public class FinancialPredictor {
    private final MLPipeline pipeline;
    private final CudaNative.LSTMResult state;
    
    public FinancialPredictor(int inputSize, int hiddenSize) {
        this.pipeline = new MLPipeline();
        this.state = new CudaNative.LSTMResult(
            new float[hiddenSize], new float[hiddenSize]);
    }
    
    public float predictNext(float[] timeSeries) {
        // 1. Java-based feature extraction
        float[] features = extractFeatures(timeSeries);
        
        // 2. CUDA-accelerated LSTM prediction
        CudaNative.LSTMResult newState = CudaNative.lstmForward(
            features, state.hNext, state.cNext,
            getWeights(), getBiases(),
            features.length, state.hNext.length);
        
        // 3. Update state
        System.arraycopy(newState.hNext, 0, state.hNext, 0, state.hNext.length);
        System.arraycopy(newState.cNext, 0, state.cNext, 0, state.cNext.length);
        
        // 4. Python post-processing
        return postProcess(newState.hNext);
    }
    
    private native float[] getWeights();
    private native float[] getBiases();
    private native float[] extractFeatures(float[] series);
    private native float postProcess(float[] hiddenState);
}