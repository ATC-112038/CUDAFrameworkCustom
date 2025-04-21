public class MedicalImaging {
    private final MLPipeline pipeline;
    
    public MedicalImaging() {
        this.pipeline = new MLPipeline();
    }
    
    public float[] processScan(float[] scanData, int width, int height) {
        // 1. Java-based DICOM parsing
        float[] normalized = normalizeDICOM(scanData);
        
        // 2. CUDA-accelerated CNN processing
        float[] features = pipeline.processImage(
            normalized, width, height, 
            MLPipeline.ProcessingLocation.CUDA_ACCELERATED);
        
        // 3. Python-based anomaly detection
        return detectAnomalies(features);
    }
    
    private native float[] normalizeDICOM(float[] dicomData);
    private native float[] detectAnomalies(float[] features);
}