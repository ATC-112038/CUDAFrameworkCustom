package com.cuda.ml;

public class CudaNative {
    static {
        System.loadLibrary("cuda_ml_jni");
    }

    // Vector operations
    public static native float[] vectorAdd(float[] a, float[] b);
    public static native float dotProduct(float[] a, float[] b);
    
    // CNN operations
    public static native float[] conv2d(
        float[] input, int[] inputShape,
        float[] filters, int[] filterShape,
        int stride, int padding);
    
    // LSTM operations
    public static native LSTMResult lstmForward(
        float[] input, float[] hPrev, float[] cPrev,
        float[] weights, float[] biases,
        int inputSize, int hiddenSize);
    
    public static class LSTMResult {
        public final float[] hNext;
        public final float[] cNext;
        
        public LSTMResult(float[] hNext, float[] cNext) {
            this.hNext = hNext;
            this.cNext = cNext;
        }
    }
}