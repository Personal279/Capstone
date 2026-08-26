package com.pes.facialparalysis.ui.screens

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pes.facialparalysis.data.AppDatabase
import com.pes.facialparalysis.data.AssessmentRecord
import com.pes.facialparalysis.data.CapturedImageHolder
import com.pes.facialparalysis.data.CapturedVideoHolder
import com.pes.facialparalysis.data.SelectedPatientHolder
import com.pes.facialparalysis.ml.ExplanationGenerator
import com.pes.facialparalysis.ml.FaceLandmarkDetector
import com.pes.facialparalysis.ml.FaiCalculator
import com.pes.facialparalysis.ml.FrameExtractor
import com.pes.facialparalysis.ml.MlpClassifier
import com.pes.facialparalysis.ml.ResNetFeatureExtractor
import com.pes.facialparalysis.ui.theme.AppColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private data class ScreenResult(
    val grade: Int? = null,
    val probabilities: Map<Int, Double> = emptyMap(),
    val error: String? = null,
    val framesUsed: Int = 0,
    val explanation: String = ""
)

@Composable
fun ResultScreen(onDone: () -> Unit) {
    val context = LocalContext.current
    var isLoading by remember { mutableStateOf(true) }
    var result by remember { mutableStateOf(ScreenResult()) }

    LaunchedEffect(Unit) {
        val imageBitmap = CapturedImageHolder.bitmap
        val videoFile = CapturedVideoHolder.videoFile

        if (imageBitmap == null && videoFile == null) {
            result = ScreenResult(error = "No image or video found. Please try again.")
            isLoading = false
            return@LaunchedEffect
        }

        val outcome = withContext(Dispatchers.Default) {
            try {
                val framesToProcess = if (videoFile != null) {
                    FrameExtractor.extract(videoFile, maxFrames = 10)
                } else {
                    listOf(imageBitmap!!)
                }

                if (framesToProcess.isEmpty()) {
                    return@withContext ScreenResult(error = "Could not read frames from video.")
                }

                val landmarkDetector = FaceLandmarkDetector(context)
                val resnetExtractor = ResNetFeatureExtractor(context)
                val mlp = MlpClassifier(context)

                val perFrameGrades = mutableListOf<Int>()
                val perFrameProbabilities = mutableListOf<Map<Int, Double>>()
                var lastBreakdown: FaiCalculator.FaiBreakdown? = null
                var validFrames = 0

                for (frame in framesToProcess) {
                    val landmarkResult = landmarkDetector.detect(frame) ?: continue
                    val landmarks = landmarkResult.faceLandmarks()[0]
                    val breakdown = FaiCalculator.computeBreakdown(landmarks, frame.width, frame.height)
                    lastBreakdown = breakdown

                    val embedding = resnetExtractor.extract(frame)

                    // 5 FAI features + 2048 ResNet features = 2053, must match scaler.json/mlp.json
                    val fused = DoubleArray(2053)
                    val faiValues = breakdown.toFeatureArray() // [eye, mouth, brow, cheek, jaw]
                    for (i in faiValues.indices) fused[i] = faiValues[i]
                    for (i in embedding.indices) fused[i + 5] = embedding[i].toDouble()

                    val prediction = mlp.predict(fused)
                    perFrameGrades.add(prediction.predictedGrade)
                    perFrameProbabilities.add(prediction.probabilities)
                    validFrames++
                }

                landmarkDetector.close()
                resnetExtractor.close()

                if (validFrames == 0 || lastBreakdown == null) {
                    return@withContext ScreenResult(error = "No face detected. Please retake with a clear frontal face.")
                }

                // Model now has 4 classes (1-4), grade 5 was dropped during training
                val avgProbabilities = (1..4).associateWith { grade ->
                    perFrameProbabilities.map { it[grade] ?: 0.0 }.average()
                }
                val finalGrade = avgProbabilities.maxByOrNull { it.value }?.key ?: perFrameGrades.first()
                val explanation = ExplanationGenerator.generate(lastBreakdown, finalGrade)

                ScreenResult(
                    grade = finalGrade,
                    probabilities = avgProbabilities,
                    framesUsed = validFrames,
                    explanation = explanation
                )
            } catch (e: Exception) {
                ScreenResult(error = "Something went wrong. Please try again.")
            }
        }

        val patientId = SelectedPatientHolder.patientId
        if (outcome.grade != null && patientId != null) {
            val gradeLabel = when (outcome.grade) {
                1 -> "Normal"
                2 -> "Mild"
                3 -> "Moderate"
                else -> "Moderately severe"
            }
            val confidence = outcome.probabilities[outcome.grade] ?: 0.0
            val record = AssessmentRecord(
                patientId = patientId,
                timestamp = System.currentTimeMillis(),
                grade = outcome.grade,
                gradeLabel = gradeLabel,
                faiValue = 0.0,
                confidence = confidence
            )
            AppDatabase.getDatabase(context).assessmentDao().insert(record)
        }

        result = outcome
        isLoading = false
        CapturedImageHolder.clear()
        CapturedVideoHolder.clear()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppColors.Background)
            .padding(24.dp)
    ) {
        Spacer(modifier = Modifier.height(20.dp))
        Text(
            text = "Analysis result",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = AppColors.TextPrimary
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = when {
                isLoading -> "Analyzing the captured images"
                result.error != null -> "We ran into a problem"
                else -> "Here's what we found"
            },
            fontSize = 13.sp,
            color = AppColors.TextSecondary
        )
        Spacer(modifier = Modifier.height(16.dp))

        Column(modifier = Modifier.weight(1f)) {
            when {
                isLoading -> LoadingState()
                result.error != null -> ErrorState(result.error!!)
                result.grade != null -> ResultState(result.grade!!, result.probabilities, result.framesUsed, result.explanation)
            }
        }

        Button(
            onClick = onDone,
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(10.dp),
            colors = ButtonDefaults.buttonColors(containerColor = AppColors.Primary)
        ) {
            Text("Back to home", fontSize = 15.sp, fontWeight = FontWeight.Medium, color = AppColors.TextOnPrimary)
        }
    }
}

@Composable
private fun LoadingState() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(color = AppColors.Primary)
            Spacer(modifier = Modifier.height(16.dp))
            Text("Analyzing...", fontSize = 15.sp, color = AppColors.TextPrimary)
        }
    }
}

@Composable
private fun ErrorState(message: String) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
            border = BorderStroke(1.dp, AppColors.Border),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp).fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Filled.Warning,
                    contentDescription = null,
                    tint = AppColors.Warning,
                    modifier = Modifier.size(40.dp)
                )
                Spacer(modifier = Modifier.height(14.dp))
                Text(
                    text = message,
                    fontSize = 15.sp,
                    color = AppColors.TextPrimary,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@Composable
private fun ResultState(grade: Int, probabilities: Map<Int, Double>, framesUsed: Int, explanation: String) {
    val gradeColor = AppColors.gradeColor(grade)
    val gradeLabel = when (grade) {
        1 -> "Normal"
        2 -> "Mild"
        3 -> "Moderate"
        else -> "Moderately severe"
    }

    // Summary card — mirrors PatientCard's avatar + text row layout
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
        border = BorderStroke(1.dp, AppColors.Border),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(gradeColor.copy(alpha = 0.12f)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "G$grade",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = gradeColor
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = gradeLabel,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Medium,
                    color = AppColors.TextPrimary
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = if (framesUsed > 1) "Averaged across $framesUsed frames" else "Analysis complete",
                    fontSize = 12.sp,
                    color = AppColors.TextSecondary
                )
            }
            Icon(
                imageVector = Icons.Filled.CheckCircle,
                contentDescription = null,
                tint = AppColors.Primary,
                modifier = Modifier.size(20.dp)
            )
        }
    }

    Spacer(modifier = Modifier.height(12.dp))

    // Explanation card
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = AppColors.PrimaryLight),
        border = BorderStroke(1.dp, AppColors.Border),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Row(modifier = Modifier.padding(16.dp)) {
            Box(
                modifier = Modifier
                    .size(28.dp)
                    .clip(CircleShape)
                    .background(AppColors.Primary.copy(alpha = 0.15f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Info,
                    contentDescription = null,
                    tint = AppColors.Primary,
                    modifier = Modifier.size(16.dp)
                )
            }
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = explanation,
                fontSize = 13.sp,
                color = AppColors.TextPrimary,
                lineHeight = 18.sp,
                modifier = Modifier.align(Alignment.CenterVertically)
            )
        }
    }

    Spacer(modifier = Modifier.height(12.dp))

    // Confidence breakdown card
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
        border = BorderStroke(1.dp, AppColors.Border),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = "Confidence by grade",
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                color = AppColors.TextPrimary
            )
            Spacer(modifier = Modifier.height(14.dp))
            probabilities.toSortedMap().forEach { (g, prob) ->
                ProbabilityRow(grade = g, probability = prob, isSelected = g == grade)
                Spacer(modifier = Modifier.height(10.dp))
            }
        }
    }
}

@Composable
private fun ProbabilityRow(grade: Int, probability: Double, isSelected: Boolean) {
    val animatedProgress by animateFloatAsState(
        targetValue = probability.toFloat(),
        animationSpec = tween(durationMillis = 600),
        label = "probability"
    )
    val barColor = if (isSelected) AppColors.Primary else AppColors.Primary.copy(alpha = 0.3f)

    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = "G$grade",
            fontSize = 12.sp,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
            color = AppColors.TextPrimary,
            modifier = Modifier.width(26.dp)
        )
        Box(
            modifier = Modifier
                .weight(1f)
                .height(7.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(AppColors.SurfaceMuted)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(animatedProgress)
                    .clip(RoundedCornerShape(4.dp))
                    .background(barColor)
            )
        }
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "${"%.0f".format(probability * 100)}%",
            fontSize = 12.sp,
            color = AppColors.TextPrimary,
            modifier = Modifier.width(34.dp)
        )
    }
}