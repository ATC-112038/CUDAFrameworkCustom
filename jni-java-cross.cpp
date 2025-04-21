#include <jni.h>
#include "com_cuda_ml_CudaNative.h"
#include "../../../cuda_kernels/include/cuda_interface.h"

JNIEXPORT jfloatArray JNICALL Java_com_cuda_ml_CudaNative_vectorAdd
  (JNIEnv *env, jclass cls, jfloatArray a, jfloatArray b) {
    
    jsize len = env->GetArrayLength(a);
    jfloat* a_data = env->GetFloatArrayElements(a, 0);
    jfloat* b_data = env->GetFloatArrayElements(b, 0);
    
    float* result = new float[len];
    cuda_vector_add(a_data, b_data, result, len);
    
    jfloatArray jresult = env->NewFloatArray(len);
    env->SetFloatArrayRegion(jresult, 0, len, result);
    
    // Cleanup
    env->ReleaseFloatArrayElements(a, a_data, 0);
    env->ReleaseFloatArrayElements(b, b_data, 0);
    delete[] result;
    
    return jresult;
}

// Similar implementations for other native methods