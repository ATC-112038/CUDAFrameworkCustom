#include "vector_ops.h"
#include <gtest/gtest.h>

class VectorOpsTest : public ::testing::Test {
protected:
    const size_t N = 1 << 20; // 1M elements
    float *a, *b, *c;
    
    void SetUp() override {
        a = VectorOperations::allocateUnifiedMemory(N);
        b = VectorOperations::allocateUnifiedMemory(N);
        c = VectorOperations::allocateUnifiedMemory(N);
        VectorOperations::fillRandom(a, N);
        VectorOperations::fillRandom(b, N);
    }
    
    void TearDown() override {
        VectorOperations::freeUnifiedMemory(a);
        VectorOperations::freeUnifiedMemory(b);
        VectorOperations::freeUnifiedMemory(c);
    }
};

TEST_F(VectorOpsTest, VectorAddition) {
    VectorOperations::unifiedVectorAdd(a, b, c, N);
    ASSERT_TRUE(VectorOperations::verifyResults(a, b, c, N));
}

// [Additional tests would follow...]