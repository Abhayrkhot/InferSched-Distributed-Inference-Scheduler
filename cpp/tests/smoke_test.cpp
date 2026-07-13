// Phase 0 smoke test: proves the build graph is sound — the core library links,
// and the generated gRPC/protobuf contract compiles and is usable.

#include <gtest/gtest.h>

#include "infersched/version.hpp"
#include "infersched.pb.h"  // generated from proto/infersched.proto

TEST(Smoke, VersionIsReported) {
  EXPECT_EQ(infersched::Version(), "0.7.0");
}

TEST(Smoke, GeneratedProtoIsUsable) {
  // Exercise the generated contract: set/read the fence used for ownership
  // fencing (partition_id, ownership_epoch, attempt_id).
  infersched::v1::Fence fence;
  fence.set_partition_id(3);
  fence.set_ownership_epoch(7);
  fence.set_attempt_id(42);

  EXPECT_EQ(fence.partition_id(), 3);
  EXPECT_EQ(fence.ownership_epoch(), 7u);
  EXPECT_EQ(fence.attempt_id(), 42u);
}

TEST(Smoke, DispatchAckStatusEnumExists) {
  // The explicit ACCEPTED/REJECTED handshake must be present.
  EXPECT_EQ(infersched::v1::DispatchAck::ACCEPTED, 0);
  EXPECT_EQ(infersched::v1::DispatchAck::REJECTED, 1);

  infersched::v1::DispatchEvent event;
  event.mutable_ack()->set_status(infersched::v1::DispatchAck::ACCEPTED);
  EXPECT_TRUE(event.has_ack());
  EXPECT_EQ(event.ack().status(), infersched::v1::DispatchAck::ACCEPTED);
}
