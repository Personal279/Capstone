package com.pes.facialparalysis.ui.screens

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import com.pes.facialparalysis.data.XaiDataHolder
import com.pes.facialparalysis.ml.ExplanationGenerator
import com.pes.facialparalysis.ml.FaceLandmarkDetector
import com.pes.facialparalysis.ml.FaiCalculator
import com.pes.facialparalysis.ml.FrameExtractor
import com.pes.facialparalysis.ml.LimeExplainer
import com.pes.facialparalysis.ml.MlpClassifier
import com.pes.facialparalysis.ml.ScanQualityChecker
import com.pes.facialparalysis.ml.XaiExplainer
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.components.ConfidenceBars
import com.pes.facialparalysis.ui.theme.components.DisclaimerBanner
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private data class ScreenResult(
    val grade: Int? = null,
    val probabilities: Map<Int, Double> = emptyMap(),
    val error: String? = null,
    val framesUsed: Int = 0,
    val explanation: String = "",
    val regions: List<FaiCalculator.RegionAsymmetry> = emptyList(),
    val xaiSummary: String = "",
    val xaiBullets: List<String> = emptyList(),
    val analyzedFrame: Bitmap? = null,
    val analyzedLandmarks: List<NormalizedLandmark>? = null,
    val limeResult: LimeExplainer.LimeResult? = null,
    val scanQuality: ScanQualityChecker.ScanQualityResult? = null
)

@Composable
fun ResultScreen(onDone: () -> Unit, onViewExplanation: () -> Unit = {}, onViewDigitalTwin: () -> Unit = {}) {
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
                val mlp = MlpClassifier(context)

                val perFrameGrades = mutableListOf<Int>()
                val perFrameProbabilities = mutableListOf<Map<Int, Double>>()
                val perFrameRegions = mutableListOf<List<FaiCalculator.RegionAsymmetry>>()
                var firstAnalyzedFrame: Bitmap? = null
                var firstAnalyzedLandmarks: List<NormalizedLandmark>? = null
                var validFrames = 0

                for (frame in framesToProcess) {
                    val landmarkResult = landmarkDetector.detect(frame) ?: continue
                    val prediction = mlp.predict(frame)
                    val landmarks = landmarkResult.faceLandmarks()[0]
                    perFrameGrades.add(prediction.predictedGrade)
                    perFrameProbabilities.add(prediction.probabilities)
                    perFrameRegions.add(FaiCalculator.computeDetailed(landmarks, frame.width, frame.height))
                    if (firstAnalyzedFrame == null) {
                        firstAnalyzedFrame = frame
                        firstAnalyzedLandmarks = landmarks
                    }
                    validFrames++
                }

                landmarkDetector.close()

                if (validFrames == 0) {
                    mlp.close()
                    return@withContext ScreenResult(error = "No face detected. Please retake with a clear frontal face.")
                }

                val avgProbabilities = (1..5).associateWith { grade ->
                    perFrameProbabilities.map { it[grade] ?: 0.0 }.average()
                }
                val finalGrade = avgProbabilities.maxByOrNull { it.value }?.key ?: perFrameGrades.first()
                val explanation = ExplanationGenerator.generate(finalGrade)

                // Real LIME explanation of the existing classifier's own output on the analyzed
                // frame — reuses the same MlpClassifier instance, no new model involved.
                val limeResult = if (firstAnalyzedFrame != null && firstAnalyzedLandmarks != null) {
                    LimeExplainer.explain(firstAnalyzedFrame, firstAnalyzedLandmarks, finalGrade, mlp)
                } else null
                mlp.close()

                val scanQuality = if (firstAnalyzedFrame != null && firstAnalyzedLandmarks != null) {
                    ScanQualityChecker.analyze(firstAnalyzedFrame, firstAnalyzedLandmarks)
                } else null

                // Average each region's asymmetry % across analyzed frames; keep the
                // smaller-side hint from whichever frame had the largest asymmetry for that region.
                val regionNames = listOf("eye", "mouth", "brow", "cheek", "jaw")
                val avgRegions = regionNames.map { regionName ->
                    val samples = perFrameRegions.map { frameRegions -> frameRegions.first { it.region == regionName } }
                    val avgPct = samples.map { it.asymmetryPercent }.average()
                    val dominantSide = samples.maxByOrNull { it.asymmetryPercent }?.smallerSide ?: "left"
                    FaiCalculator.RegionAsymmetry(regionName, avgPct, dominantSide)
                }
                val xai = XaiExplainer.explain(avgRegions)

                ScreenResult(
                    grade = finalGrade,
                    probabilities = avgProbabilities,
                    framesUsed = validFrames,
                    explanation = explanation,
                    regions = avgRegions,
                    xaiSummary = xai.summarySentence,
                    xaiBullets = xai.bulletLabels,
                    analyzedFrame = firstAnalyzedFrame,
                    analyzedLandmarks = firstAnalyzedLandmarks,
                    limeResult = limeResult,
                    scanQuality = scanQuality
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
                4 -> "Moderately severe"
                else -> "Severe"
            }
            val confidence = outcome.probabilities[outcome.grade] ?: 0.0
            fun pct(region: String) = outcome.regions.firstOrNull { it.region == region }?.asymmetryPercent ?: 0.0
            val record = AssessmentRecord(
                patientId = patientId,
                timestamp = System.currentTimeMillis(),
                grade = outcome.grade,
                gradeLabel = gradeLabel,
                faiValue = if (outcome.regions.isNotEmpty()) outcome.regions.map { it.asymmetryPercent }.average() else 0.0,
                confidence = confidence,
                eyeAsymmetry = pct("eye"),
                mouthAsymmetry = pct("mouth"),
                browAsymmetry = pct("brow"),
                cheekAsymmetry = pct("cheek"),
                jawAsymmetry = pct("jaw"),
                topRegionsCsv = outcome.xaiBullets.joinToString(","),
                xaiExplanation = outcome.xaiSummary
            )
            AppDatabase.getDatabase(context).assessmentDao().insert(record)
        }

        // Hand the analyzed frame + landmarks to the XAI screen's transient holder rather than
        // recycling it immediately — it's still needed for "View AI Explanation". It's recycled
        // exactly once, when the user leaves this result flow (see onDone below).
        XaiDataHolder.bitmap = outcome.analyzedFrame
        XaiDataHolder.landmarks = outcome.analyzedLandmarks
        XaiDataHolder.regions = outcome.regions
        XaiDataHolder.grade = outcome.grade
        XaiDataHolder.probabilities = outcome.probabilities
        XaiDataHolder.summarySentence = outcome.xaiSummary
        XaiDataHolder.bulletLabels = outcome.xaiBullets
        XaiDataHolder.limeResult = outcome.limeResult
        XaiDataHolder.scanQuality = outcome.scanQuality

        result = outcome
        isLoading = false
        if (outcome.analyzedFrame != null) {
            CapturedImageHolder.bitmap = null // ownership moved to XaiDataHolder; do not recycle here
        } else {
            CapturedImageHolder.clear() // analysis failed before producing a frame to hand off
        }
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

        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
        ) {
            when {
                isLoading -> LoadingState()
                result.error != null -> ErrorState(result.error!!)
                result.grade != null -> ResultState(
                    grade = result.grade!!,
                    probabilities = result.probabilities,
                    framesUsed = result.framesUsed,
                    explanation = result.explanation,
                    onViewExplanation = onViewExplanation,
                    onViewDigitalTwin = onViewDigitalTwin
                )
            }
        }

        Button(
            onClick = {
                XaiDataHolder.clear()
                onDone()
            },
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
private fun ResultState(
    grade: Int,
    probabilities: Map<Int, Double>,
    framesUsed: Int,
    explanation: String,
    onViewExplanation: () -> Unit,
    onViewDigitalTwin: () -> Unit
) {
    val gradeColor = AppColors.gradeColor(grade)
    val gradeLabel = when (grade) {
        1 -> "Normal"
        2 -> "Mild"
        3 -> "Moderate"
        4 -> "Moderately severe"
        else -> "Severe"
    }

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

    ConfidenceBars(probabilities = probabilities, selectedGrade = grade)

    Spacer(modifier = Modifier.height(12.dp))

    OutlinedButton(
        onClick = onViewExplanation,
        modifier = Modifier.fillMaxWidth().height(48.dp),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, AppColors.Border),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.Primary)
    ) {
        Icon(Icons.Filled.Info, contentDescription = null, modifier = Modifier.size(16.dp))
        Spacer(modifier = Modifier.width(8.dp))
        Text("View AI Explanation", fontSize = 14.sp, fontWeight = FontWeight.Medium)
    }

    Spacer(modifier = Modifier.height(10.dp))

    OutlinedButton(
        onClick = onViewDigitalTwin,
        modifier = Modifier.fillMaxWidth().height(48.dp),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, AppColors.Border),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.TextPrimary)
    ) {
        Text("View Digital Twin", fontSize = 14.sp, fontWeight = FontWeight.Medium)
    }

    Spacer(modifier = Modifier.height(12.dp))

    DisclaimerBanner()
}