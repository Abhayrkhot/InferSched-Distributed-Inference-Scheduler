#include "infersched/durable/durable_store.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

#include <pqxx/pqxx>

namespace infersched::durable {
namespace {

std::int64_t CheckedInt64(const std::uint64_t value) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("fence value exceeds PostgreSQL BIGINT");
  }
  return static_cast<std::int64_t>(value);
}

std::uint64_t CheckedUint64(const std::int64_t value) {
  if (value < 0) {
    throw std::runtime_error("negative durable counter");
  }
  return static_cast<std::uint64_t>(value);
}

std::string BytesToString(const pqxx::bytes& bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

}  // namespace

class DurableStore::Impl {
 public:
  explicit Impl(std::string connection_string)
      : connection(std::move(connection_string)) {}

  mutable pqxx::connection connection;
};

DurableStore::DurableStore(std::string connection_string)
    : impl_(std::make_unique<Impl>(std::move(connection_string))) {}
DurableStore::~DurableStore() = default;
DurableStore::DurableStore(DurableStore&&) noexcept = default;
DurableStore& DurableStore::operator=(DurableStore&&) noexcept = default;

void DurableStore::Migrate() {
  pqxx::work transaction(impl_->connection);
  transaction.exec(R"sql(
    CREATE TABLE IF NOT EXISTS partition_ownership (
      partition_id INTEGER PRIMARY KEY,
      ownership_epoch BIGINT NOT NULL CHECK (ownership_epoch > 0),
      router_id TEXT NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS inference_requests (
      request_id TEXT PRIMARY KEY,
      state TEXT NOT NULL DEFAULT 'RECEIVED',
      state_version BIGINT NOT NULL DEFAULT 0,
      source_partition INTEGER NOT NULL,
      source_offset BIGINT NOT NULL,
      raw_payload BYTEA NOT NULL,
      ownership_epoch BIGINT,
      engine_id TEXT,
      attempt_id BIGINT NOT NULL DEFAULT 0,
      terminal_result TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE (source_partition, source_offset)
    );
    CREATE TABLE IF NOT EXISTS poison_messages (
      source_partition INTEGER NOT NULL,
      source_offset BIGINT NOT NULL,
      raw_payload BYTEA NOT NULL,
      error TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (source_partition, source_offset)
    );
    CREATE TABLE IF NOT EXISTS outbox (
      id BIGSERIAL PRIMARY KEY,
      topic TEXT NOT NULL,
      message_key TEXT NOT NULL,
      payload TEXT NOT NULL,
      source_partition INTEGER NOT NULL,
      source_offset BIGINT NOT NULL,
      published_at TIMESTAMPTZ,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      UNIQUE (topic, message_key)
    );
    CREATE INDEX IF NOT EXISTS outbox_unpublished_idx
      ON outbox (id) WHERE published_at IS NULL;
    -- IsSourcePublished runs once per completed Kafka offset; without this it
    -- is a sequential scan whose cost grows with every result ever written.
    CREATE INDEX IF NOT EXISTS outbox_source_idx
      ON outbox (source_partition, source_offset);
    -- RecoverUnresolved on every partition assignment: only scan live work.
    CREATE INDEX IF NOT EXISTS inference_requests_unresolved_idx
      ON inference_requests (source_partition, source_offset)
      WHERE state NOT IN ('COMPLETED', 'FAILED_TERMINAL', 'CANCELLED', 'TIMED_OUT');
    DO $$ BEGIN
      IF (SELECT data_type = 'text' FROM information_schema.columns
           WHERE table_name = 'inference_requests' AND column_name = 'raw_payload') THEN
        ALTER TABLE inference_requests ALTER COLUMN raw_payload TYPE BYTEA
          USING convert_to(raw_payload, 'UTF8');
      END IF;
      IF (SELECT data_type = 'text' FROM information_schema.columns
           WHERE table_name = 'poison_messages' AND column_name = 'raw_payload') THEN
        ALTER TABLE poison_messages ALTER COLUMN raw_payload TYPE BYTEA
          USING convert_to(raw_payload, 'UTF8');
      END IF;
    END $$;
  )sql").no_rows();
  transaction.commit();
}

std::uint64_t DurableStore::AcquirePartition(const std::int32_t partition_id,
                                              const std::string_view router_id) {
  pqxx::work transaction(impl_->connection);
  const pqxx::row row = transaction.exec(R"sql(
    INSERT INTO partition_ownership (partition_id, ownership_epoch, router_id)
    VALUES ($1, 1, $2)
    ON CONFLICT (partition_id) DO UPDATE
      SET ownership_epoch = partition_ownership.ownership_epoch + 1,
          router_id = EXCLUDED.router_id, updated_at = now()
    RETURNING ownership_epoch
  )sql", pqxx::params{partition_id, router_id}).one_row();
  transaction.commit();
  return CheckedUint64(row[0].as<std::int64_t>());
}

bool DurableStore::Ingest(const std::string_view request_id,
                          const std::int32_t source_partition,
                          const std::int64_t source_offset,
                          const std::string_view raw_payload) {
  pqxx::work transaction(impl_->connection);
  const pqxx::result result = transaction.exec(R"sql(
    INSERT INTO inference_requests
      (request_id, source_partition, source_offset, raw_payload)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT DO NOTHING
  )sql", pqxx::params{request_id, source_partition, source_offset,
                       pqxx::binary_cast(raw_payload)});
  transaction.commit();
  return result.affected_rows() == 1;
}

std::vector<IngestStatus> DurableStore::IngestBatch(const std::span<const IngestRow> rows) {
  std::vector<IngestStatus> statuses;
  statuses.reserve(rows.size());
  if (rows.empty()) {
    return statuses;
  }
  pqxx::work transaction(impl_->connection);
  for (const IngestRow& row : rows) {
    const pqxx::result inserted = transaction.exec(R"sql(
      INSERT INTO inference_requests
        (request_id, source_partition, source_offset, raw_payload)
      VALUES ($1, $2, $3, $4)
      ON CONFLICT DO NOTHING
    )sql", pqxx::params{row.request_id, row.source_partition, row.source_offset,
                         pqxx::binary_cast(row.raw_payload)});
    IngestStatus status{.inserted = inserted.affected_rows() == 1};
    if (!status.inserted) {
      const pqxx::result state = transaction.exec(R"sql(
        SELECT state IN ('COMPLETED', 'FAILED_TERMINAL', 'CANCELLED', 'TIMED_OUT')
          FROM inference_requests WHERE request_id = $1
      )sql", pqxx::params{row.request_id});
      status.terminal = !state.empty() && state[0][0].as<bool>();
    }
    statuses.push_back(status);
  }
  transaction.commit();
  return statuses;
}

std::optional<Assignment> DurableStore::AssignAttempt(
    const std::string_view request_id, const std::int32_t partition_id,
    const std::uint64_t ownership_epoch, const std::string_view engine_id) {
  pqxx::work transaction(impl_->connection);
  const pqxx::result result = transaction.exec(R"sql(
    UPDATE inference_requests AS request
       SET state = 'DISPATCHED', state_version = state_version + 1,
           ownership_epoch = $3, engine_id = $4, attempt_id = attempt_id + 1,
           updated_at = now()
      FROM partition_ownership AS owner
     WHERE request.request_id = $1
       AND request.source_partition = $2
       AND owner.partition_id = $2
       AND owner.ownership_epoch = $3
       AND request.state NOT IN
         ('COMPLETED', 'FAILED_TERMINAL', 'CANCELLED', 'TIMED_OUT')
    RETURNING request.attempt_id, request.state_version
  )sql", pqxx::params{request_id, partition_id, CheckedInt64(ownership_epoch),
                       engine_id});
  if (result.empty()) {
    transaction.commit();
    return std::nullopt;
  }
  const Assignment assignment{
      .fence = Fence{.partition_id = partition_id,
                     .ownership_epoch = ownership_epoch,
                     .attempt_id = CheckedUint64(result[0][0].as<std::int64_t>()),
                     .engine_id = std::string(engine_id)},
      .state_version = CheckedUint64(result[0][1].as<std::int64_t>())};
  transaction.commit();
  return assignment;
}

bool DurableStore::Transition(const std::string_view request_id,
                              const std::uint64_t expected_state_version,
                              const std::string_view expected_state,
                              const std::string_view next_state) {
  pqxx::work transaction(impl_->connection);
  const pqxx::result updated = transaction.exec(R"sql(
    UPDATE inference_requests
       SET state = $4, state_version = state_version + 1, updated_at = now()
     WHERE request_id = $1 AND state_version = $2 AND state = $3
       AND state NOT IN ('COMPLETED', 'FAILED_TERMINAL', 'CANCELLED', 'TIMED_OUT')
  )sql", pqxx::params{request_id, CheckedInt64(expected_state_version),
                       expected_state, next_state});
  transaction.commit();
  return updated.affected_rows() == 1;
}

bool DurableStore::Finalize(const std::string_view request_id, const Fence& fence,
                            const std::string_view result_payload) {
  pqxx::work transaction(impl_->connection);
  const pqxx::result updated = transaction.exec(R"sql(
    UPDATE inference_requests
       SET state = 'COMPLETED', state_version = state_version + 1,
           terminal_result = $6, updated_at = now()
     WHERE request_id = $1 AND source_partition = $2
       AND ownership_epoch = $3 AND attempt_id = $4 AND engine_id = $5
       AND state = 'DISPATCHED'
    RETURNING source_partition, source_offset
  )sql", pqxx::params{request_id, fence.partition_id,
                       CheckedInt64(fence.ownership_epoch),
                       CheckedInt64(fence.attempt_id), fence.engine_id,
                       result_payload});
  if (updated.empty()) {
    transaction.commit();
    return false;
  }
  transaction.exec(R"sql(
    INSERT INTO outbox
      (topic, message_key, payload, source_partition, source_offset)
    VALUES ('inference.results', $1, $2, $3, $4)
    ON CONFLICT (topic, message_key) DO NOTHING
  )sql", pqxx::params{request_id, result_payload,
                       updated[0][0].as<std::int32_t>(),
                       updated[0][1].as<std::int64_t>()}).no_rows();
  transaction.commit();
  return true;
}

bool DurableStore::FailTerminal(const std::string_view request_id,
                                const std::int32_t partition_id,
                                const std::uint64_t ownership_epoch,
                                const std::string_view result_payload) {
  pqxx::work transaction(impl_->connection);
  const pqxx::result updated = transaction.exec(R"sql(
    UPDATE inference_requests AS request
       SET state = 'FAILED_TERMINAL', state_version = request.state_version + 1,
           terminal_result = $4, updated_at = now()
      FROM partition_ownership AS owner
     WHERE request.request_id = $1
       AND request.source_partition = $2
       AND owner.partition_id = $2
       AND owner.ownership_epoch = $3
       AND request.state NOT IN
         ('COMPLETED', 'FAILED_TERMINAL', 'CANCELLED', 'TIMED_OUT')
    RETURNING request.source_partition, request.source_offset
  )sql", pqxx::params{request_id, partition_id, CheckedInt64(ownership_epoch),
                       result_payload});
  if (updated.empty()) {
    transaction.commit();
    return false;
  }
  transaction.exec(R"sql(
    INSERT INTO outbox
      (topic, message_key, payload, source_partition, source_offset)
    VALUES ('inference.results', $1, $2, $3, $4)
    ON CONFLICT (topic, message_key) DO NOTHING
  )sql", pqxx::params{request_id, result_payload,
                       updated[0][0].as<std::int32_t>(),
                       updated[0][1].as<std::int64_t>()}).no_rows();
  transaction.commit();
  return true;
}

bool DurableStore::RecordPoison(const std::int32_t source_partition,
                                const std::int64_t source_offset,
                                const std::string_view raw_payload,
                                const std::string_view error) {
  pqxx::work transaction(impl_->connection);
  const pqxx::result inserted = transaction.exec(R"sql(
    INSERT INTO poison_messages
      (source_partition, source_offset, raw_payload, error)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT DO NOTHING
  )sql", pqxx::params{source_partition, source_offset,
                       pqxx::binary_cast(raw_payload),
                       error});
  if (inserted.affected_rows() == 1) {
    const std::string key = std::to_string(source_partition) + ":" +
                            std::to_string(source_offset);
    transaction.exec(R"sql(
      INSERT INTO outbox
        (topic, message_key, payload, source_partition, source_offset)
      VALUES ('inference.dlq', $1, encode($2, 'hex'), $3, $4)
      ON CONFLICT (topic, message_key) DO NOTHING
    )sql", pqxx::params{
               key, pqxx::binary_cast(raw_payload),
               source_partition, source_offset})
        .no_rows();
  }
  transaction.commit();
  return inserted.affected_rows() == 1;
}

std::vector<OutboxRecord> DurableStore::FetchUnpublished(
    const std::size_t limit) const {
  pqxx::read_transaction transaction(impl_->connection);
  const pqxx::result rows = transaction.exec(R"sql(
    SELECT id, topic, message_key, payload, source_partition, source_offset
      FROM outbox WHERE published_at IS NULL ORDER BY id LIMIT $1
  )sql", pqxx::params{limit});
  std::vector<OutboxRecord> records;
  records.reserve(static_cast<std::size_t>(rows.size()));
  for (const auto row : rows) {
    records.push_back(OutboxRecord{
        .id = row[0].as<std::int64_t>(), .topic = row[1].as<std::string>(),
        .message_key = row[2].as<std::string>(),
        .payload = row[3].as<std::string>(),
        .source_partition = row[4].as<std::int32_t>(),
        .source_offset = row[5].as<std::int64_t>()});
  }
  return records;
}

void DurableStore::MarkPublished(const std::int64_t outbox_id) {
  pqxx::work transaction(impl_->connection);
  transaction.exec(
      "UPDATE outbox SET published_at = COALESCE(published_at, now()) WHERE id = $1",
      pqxx::params{outbox_id}).no_rows();
  transaction.commit();
}

void DurableStore::MarkPublishedMany(const std::span<const std::int64_t> outbox_ids) {
  if (outbox_ids.empty()) {
    return;
  }
  std::string ids;
  for (const std::int64_t id : outbox_ids) {
    if (!ids.empty()) {
      ids.push_back(',');
    }
    ids += std::to_string(id);
  }
  pqxx::work transaction(impl_->connection);
  transaction.exec(R"sql(
    UPDATE outbox SET published_at = COALESCE(published_at, now())
     WHERE id = ANY(string_to_array($1, ',')::BIGINT[])
  )sql", pqxx::params{ids}).no_rows();
  transaction.commit();
}

std::vector<RecoveredRequest> DurableStore::RecoverUnresolved() const {
  pqxx::read_transaction transaction(impl_->connection);
  const pqxx::result rows = transaction.exec(R"sql(
    SELECT request_id, state, source_partition, source_offset, state_version,
           raw_payload
      FROM inference_requests
     WHERE state NOT IN ('COMPLETED', 'FAILED_TERMINAL', 'CANCELLED', 'TIMED_OUT')
     ORDER BY source_partition, source_offset
  )sql");
  std::vector<RecoveredRequest> requests;
  requests.reserve(static_cast<std::size_t>(rows.size()));
  for (const auto row : rows) {
    requests.push_back(RecoveredRequest{
        .request_id = row[0].as<std::string>(), .state = row[1].as<std::string>(),
        .source_partition = row[2].as<std::int32_t>(),
        .source_offset = row[3].as<std::int64_t>(),
        .state_version = CheckedUint64(row[4].as<std::int64_t>()),
        .raw_payload = BytesToString(row[5].as<pqxx::bytes>())});
  }
  return requests;
}

bool DurableStore::IsTerminal(const std::string_view request_id) const {
  pqxx::read_transaction transaction(impl_->connection);
  const pqxx::result rows = transaction.exec(R"sql(
    SELECT state IN ('COMPLETED', 'FAILED_TERMINAL', 'CANCELLED', 'TIMED_OUT')
      FROM inference_requests WHERE request_id = $1
  )sql", pqxx::params{request_id});
  return !rows.empty() && rows[0][0].as<bool>();
}

bool DurableStore::IsSourcePublished(const std::int32_t source_partition,
                                     const std::int64_t source_offset) const {
  pqxx::read_transaction transaction(impl_->connection);
  const pqxx::result rows = transaction.exec(R"sql(
    SELECT bool_and(published_at IS NOT NULL)
      FROM outbox
     WHERE source_partition = $1 AND source_offset = $2
  )sql", pqxx::params{source_partition, source_offset});
  return !rows.empty() && !rows[0][0].is_null() && rows[0][0].as<bool>();
}

std::vector<bool> DurableStore::SourcesPublished(
    const std::span<const std::pair<std::int32_t, std::int64_t>> sources) const {
  std::vector<bool> published(sources.size(), false);
  if (sources.empty()) {
    return published;
  }
  // Integers only, so inlining them as a VALUES list is injection-safe.
  std::string values;
  for (std::size_t index = 0; index < sources.size(); ++index) {
    values += index == 0 ? "(" : ",(";
    values += std::to_string(index) + "::int," + std::to_string(sources[index].first) +
              "::int," + std::to_string(sources[index].second) + "::bigint)";
  }
  pqxx::read_transaction transaction(impl_->connection);
  const pqxx::result rows = transaction.exec(
      "SELECT source.position, bool_and(outbox.published_at IS NOT NULL) "
      "FROM (VALUES " + values + ") AS source(position, partition_id, offset_id) "
      "LEFT JOIN outbox ON outbox.source_partition = source.partition_id "
      "AND outbox.source_offset = source.offset_id "
      "GROUP BY source.position");
  for (const auto row : rows) {
    const auto position = static_cast<std::size_t>(row[0].as<int>());
    published[position] = !row[1].is_null() && row[1].as<bool>();
  }
  return published;
}

}  // namespace infersched::durable
