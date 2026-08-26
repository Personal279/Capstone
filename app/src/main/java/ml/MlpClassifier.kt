package com.pes.facialparalysis.ml

import android.content.Context
import org.json.JSONObject
import kotlin.math.exp

class MlpClassifier(context: Context) {

    private val mean: DoubleArray
    private val scale: DoubleArray
    private val coefs: Array<Array<DoubleArray>>   // [layer][input][output]
    private val intercepts: Array<DoubleArray>       // [layer][output]
    private val classes: IntArray

    init {
        val scalerJson = JSONObject(
            context.assets.open("scaler.json").bufferedReader().use { it.readText() }
        )
        mean = jsonArrayToDoubleArray(scalerJson.getJSONArray("mean"))
        scale = jsonArrayToDoubleArray(scalerJson.getJSONArray("scale"))

        val mlpJson = JSONObject(
            context.assets.open("mlp.json").bufferedReader().use { it.readText() }
        )

        val coefsJsonArr = mlpJson.getJSONArray("coefs")
        coefs = Array(coefsJsonArr.length()) { layerIdx ->
            val layerArr = coefsJsonArr.getJSONArray(layerIdx)
            Array(layerArr.length()) { rowIdx ->
                jsonArrayToDoubleArray(layerArr.getJSONArray(rowIdx))
            }
        }

        val interceptsJsonArr = mlpJson.getJSONArray("intercepts")
        intercepts = Array(interceptsJsonArr.length()) { layerIdx ->
            jsonArrayToDoubleArray(interceptsJsonArr.getJSONArray(layerIdx))
        }

        val classesJsonArr = mlpJson.getJSONArray("classes")
        classes = IntArray(classesJsonArr.length()) { classesJsonArr.getInt(it) }
    }

    private fun jsonArrayToDoubleArray(arr: org.json.JSONArray): DoubleArray {
        return DoubleArray(arr.length()) { arr.getDouble(it) }
    }

    /** Scales input using StandardScaler: (x - mean) / scale */
    private fun scaleInput(input: DoubleArray): DoubleArray {
        return DoubleArray(input.size) { i -> (input[i] - mean[i]) / scale[i] }
    }

    private fun relu(x: DoubleArray): DoubleArray = DoubleArray(x.size) { if (x[it] > 0) x[it] else 0.0 }

    private fun softmax(x: DoubleArray): DoubleArray {
        val max = x.maxOrNull() ?: 0.0
        val exps = DoubleArray(x.size) { exp(x[it] - max) }
        val sum = exps.sum()
        return DoubleArray(x.size) { exps[it] / sum }
    }

    /** Dense layer: output[j] = sum_i(input[i] * weight[i][j]) + bias[j] */
    private fun dense(input: DoubleArray, weights: Array<DoubleArray>, bias: DoubleArray): DoubleArray {
        val outputSize = bias.size
        val output = DoubleArray(outputSize)
        for (j in 0 until outputSize) {
            var sum = 0.0
            for (i in input.indices) {
                sum += input[i] * weights[i][j]
            }
            output[j] = sum + bias[j]
        }
        return output
    }

    data class PredictionResult(val predictedGrade: Int, val probabilities: Map<Int, Double>)

    /** fusedInput must be size 2053: [eye, mouth, brow, cheek, jaw, ...2048 resnet embedding values] */
    fun predict(fusedInput: DoubleArray): PredictionResult {
        require(fusedInput.size == mean.size) {
            "Input size ${fusedInput.size} does not match expected ${mean.size}"
        }

        var activation = scaleInput(fusedInput)

        // Hidden layers (all but last) use ReLU
        for (layerIdx in 0 until coefs.size - 1) {
            activation = relu(dense(activation, coefs[layerIdx], intercepts[layerIdx]))
        }

        // Final layer uses softmax
        val logits = dense(activation, coefs.last(), intercepts.last())
        val probs = softmax(logits)

        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val predictedGrade = classes[maxIdx]

        val probMap = classes.indices.associate { classes[it] to probs[it] }

        return PredictionResult(predictedGrade, probMap)
    }
}