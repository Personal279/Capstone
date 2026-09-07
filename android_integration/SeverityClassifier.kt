package com.pes.facialparalysis.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Replaces the FaiCalculator + ResNetFeatureExtractor + MlpClassifier pipeline.
 *
 * That pipeline was built for the FAI-fusion architecture (image embedding + 5
 * FAI values -> sklearn MLP). It was found during review that the FAI values
 * fed to the classifier are exactly what the labeling pipeline used to assign
 * each training label, letting the classifier reconstruct the labeling rule
 * instead of learning from the image -- so it was dropped in favor of the
 * simpler, review-validated model this class loads: a single self-contained
 * network, image in, 5 calibrated class probabilities out (softmax already
 * applied inside the model -- no separate MLP step, no FAI, no fusion).
 *
 * Preprocessing (224x224, NHWC, ImageNet mean/std) is copied verbatim from
 * ResNetFeatureExtractor.kt's bitmapToInputBuffer() -- unchanged, since the
 * training-time normalization is identical for this model.
 */
class SeverityClassifier(context: Context) {

    data class Prediction(
        val predictedGrade: Int,           // 1-5
        val probabilities: Map<Int, Double> // grade (1-5) -> probability
    )

    private val interpreter: Interpreter

    companion object {
        private const val INPUT_SIZE = 224
        private const val NUM_CLASSES = 5
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Use the float16 asset by default -- verified during conversion to
        // differ from the float32 PyTorch output by <0.0001 per class, at
        // half the file size (44.9MB vs 89.6MB). Switch the filename below
        // if you'd rather ship float32.
        private const val MODEL_FILE = "severity_model_float16.tflite"
    }

    init {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        interpreter = Interpreter(modelBuffer)
    }

    /** Resizes bitmap to 224x224, normalizes with ImageNet mean/std, runs
     * inference, returns the predicted grade (1-5) and per-grade probabilities. */
    fun predict(bitmap: Bitmap): Prediction {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = bitmapToInputBuffer(resized)

        // Model output: [1, 5] softmax probabilities, in grade order 1..5.
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        interpreter.run(inputBuffer, output)

        val probs = output[0]
        var bestIdx = 0
        for (i in 1 until NUM_CLASSES) {
            if (probs[i] > probs[bestIdx]) bestIdx = i
        }

        val probabilityMap = (0 until NUM_CLASSES).associate { i -> (i + 1) to probs[i].toDouble() }
        return Prediction(predictedGrade = bestIdx + 1, probabilities = probabilityMap)
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        // NHWC format: [1, 224, 224, 3] -- same convention as
        // ResNetFeatureExtractor.kt's onnx2tf-converted model.
        val buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat((r - MEAN[0]) / STD[0])
            buffer.putFloat((g - MEAN[1]) / STD[1])
            buffer.putFloat((b - MEAN[2]) / STD[2])
        }
        buffer.rewind()
        return buffer
    }

    fun close() {
        interpreter.close()
    }
}
