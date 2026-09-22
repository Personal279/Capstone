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
import androidx.compose.material.icons.filled.FactCheck
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material.icons.filled.Visibility
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
import com.pes.facialparalysis.ui.theme.components.ClinicalCard
import com.pes.facialparalysis.ui.theme.components.ClinicalDivider
import com.pes.facialparalysis.ui.theme.components.ClinicalMetricRow
import com.pes.facialparalysis.ui.theme.components.ConfidenceBars
import com.pes.facialparalysis.ui.theme.components.DisclaimerBanner
import com.pes.facialparalysis.ui.theme.components.EyebrowLabel
import com.pes.facialparalysis.ui.theme.components.FaceMeshVisualization
import com.pes.facialparalysis.ui.theme.components.LandmarkAsymmetryOverlay
import com.pes.facialparalysis.ui.theme.components.PrimaryButton
import com.pes.facialparalysis.ui.theme.components.ScreenHeading
import com.pes.facialparalysis.ui.theme.components.SecondaryButton
import com.pes.facialparalysis.ui.theme.components.StageChecklistRow
import com.pes.facialparalysis.ui.theme.components.StageState
import com.pes.facialparalysis.ui.theme.components.StatusIndicator
import com.pes.facialparalysis.ui.theme.components.clinicalBackgroundBrush
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

/** Visual stage of the (single, uninterrupted) analysis pipeline — purely a UI-progress label
 * driven by real checkpoints in that pipeline; it never gates or reorders the underlying work. */
private enum class ProcessingStage { LANDMARKS, SYMMETRY, AI_REVIEW, DONE }

@Composable
fun ResultScreen(onDone: () -> Unit, onViewExplanation: () -> Unit = {}, onViewDigitalTwin: () -> Unit = {}) {
    val context = LocalContext.current
    var isLoading by remember { mutableStateOf(true) }
    var result by remember { mutableStateOf(ScreenResult()) }
    var stage by remember { mutableStateOf(ProcessingStage.LANDMARKS) }
    var liveFrame by remember { mutableStateOf<Bitmap?>(null) }
    var liveLandmarks by remember { mutableStateOf<List<NormalizedLandmark>?>(null) }
    var liveRegions by remember { mutableStateOf<List<FaiCalculator.RegionAsymmetry>>(emptyList()) }

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
                    val landmarks = landmarkResult.faceLandmarks()[0]
                    val isFirstValidFrame = validFrames == 0
                    if (isFirstValidFrame) {
                        // UI-only checkpoint: surface the real landmarks for this frame as soon as
                        // they're available. No inference call, no reordering of any ML step.
                        withContext(Dispatchers.Main) {
                            liveFrame = frame
                            liveLandmarks = landmarks
                            stage = ProcessingStage.LANDMARKS
                        }
                    }

                    val prediction = mlp.predict(frame)
                    perFrameGrades.add(prediction.predictedGrade)
                    perFrameProbabilities.add(prediction.probabilities)
                    val frameRegions = FaiCalculator.computeDetailed(landmarks, frame.width, frame.height)
                    perFrameRegions.add(frameRegions)

                    if (isFirstValidFrame) {
                        firstAnalyzedFrame = frame
                        firstAnalyzedLandmarks = landmarks
                        // UI-only checkpoint: this frame's real FAI/asymmetry values are ready.
                        withContext(Dispatchers.Main) {
                            liveRegions = frameRegions
                            stage = ProcessingStage.SYMMETRY
                        }
                    }
                    validFrames++
                }

                landmarkDetector.close()

                if (validFrames == 0) {
                    mlp.close()
                    return@withContext ScreenResult(error = "No face detected. Please retake with a clear frontal face.")
                }

                withContext(Dispatchers.Main) { stage = ProcessingStage.AI_REVIEW }

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
        stage = ProcessingStage.DONE
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
            .background(clinicalBackgroundBrush())
            .padding(24.dp)
    ) {
        Spacer(modifier = Modifier.windowInsetsTopHeight(WindowInsets.statusBars))
        Spacer(modifier = Modifier.height(12.dp))

        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
        ) {
            when {
                result.error != null -> ErrorState(result.error!!)
                stage == ProcessingStage.LANDMARKS -> LandmarkMappingState(liveLandmarks)
                stage == ProcessingStage.SYMMETRY -> SymmetryAnalysisState(liveLandmarks, liveRegions)
                stage == ProcessingStage.AI_REVIEW -> AiReviewState()
                result.grade != null -> ResultState(
                    grade = result.grade!!,
                    probabilities = result.probabilities,
                    framesUsed = result.framesUsed,
                    explanation = result.explanation,
                    onViewExplanation = onViewExplanation,
                    onViewDigitalTwin = onViewDigitalTwin
                )
                else -> LandmarkMappingState(liveLandmarks)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
        SecondaryButton(
            text = "Back to home",
            onClick = {
                XaiDataHolder.clear()
                onDone()
            },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.windowInsetsBottomHeight(WindowInsets.navigationBars))
    }
}

@Composable
private fun LandmarkMappingState(landmarks: List<NormalizedLandmark>?) {
    Column {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            EyebrowLabel("Live landmark mapping")
            StatusIndicator("Tracking")
        }
        Spacer(modifier = Modifier.height(10.dp))
        ScreenHeading("Mapping facial movement.")
        Spacer(modifier = Modifier.height(20.dp))
        ClinicalCard(contentPadding = PaddingValues(0.dp)) {
            FaceMeshVisualization(
                landmarks = landmarks,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(260.dp)
            )
        }
        Spacer(modifier = Modifier.height(16.dp))
        ClinicalCard {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Visibility, contentDescription = null, tint = AppColors.Primary, modifier = Modifier.size(20.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Column {
                    Text(
                        text = "${landmarks?.size ?: 0} landmarks localized",
                        style = MaterialTheme.typography.titleMedium,
                        color = AppColors.TextPrimary
                    )
                    Text(
                        text = "Stable facial geometry detected",
                        style = MaterialTheme.typography.bodySmall,
                        color = AppColors.TextSecondary
                    )
                }
            }
        }
    }
}

@Composable
private fun SymmetryAnalysisState(
    landmarks: List<NormalizedLandmark>?,
    regions: List<FaiCalculator.RegionAsymmetry>
) {
    Column {
        EyebrowLabel("Symmetry analysis")
        Spacer(modifier = Modifier.height(10.dp))
        ScreenHeading("A balanced clinical view.")
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            "The centerline compares resting facial position across key regions.",
            style = MaterialTheme.typography.bodyMedium,
            color = AppColors.TextSecondary
        )
        Spacer(modifier = Modifier.height(20.dp))
        ClinicalCard(contentPadding = PaddingValues(0.dp)) {
            Box {
                if (landmarks != null) {
                    LandmarkAsymmetryOverlay(
                        landmarks = landmarks,
                        regions = regions,
                        modifier = Modifier.fillMaxWidth().height(260.dp)
                    )
                } else {
                    Spacer(modifier = Modifier.fillMaxWidth().height(260.dp))
                }
            }
        }
        Spacer(modifier = Modifier.height(16.dp))
        ClinicalCard {
            regions.forEachIndexed { index, region ->
                ClinicalMetricRow(
                    label = "${region.region.replaceFirstChar { it.uppercase() }} asymmetry",
                    value = "%.1f%%".format(region.asymmetryPercent),
                    valueColor = if (region.asymmetryPercent > 15.0) AppColors.AccentRose else AppColors.AccentAmber
                )
                if (index != regions.lastIndex) {
                    Spacer(modifier = Modifier.height(10.dp))
                    ClinicalDivider()
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }
        }
    }
}

@Composable
private fun AiReviewState() {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
        EyebrowLabel("Clinical AI review", modifier = Modifier.align(Alignment.Start))
        Spacer(modifier = Modifier.height(28.dp))
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(CircleShape)
                .background(AppColors.PrimaryLight),
            contentAlignment = Alignment.Center
        ) {
            CircularProgressIndicator(color = AppColors.Primary, strokeWidth = 2.5.dp, modifier = Modifier.size(56.dp))
            Icon(Icons.Filled.Psychology, contentDescription = null, tint = AppColors.Primary, modifier = Modifier.size(28.dp))
        }
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            "Interpreting movement patterns.",
            style = MaterialTheme.typography.displayMedium,
            color = AppColors.TextPrimary,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "Combining symmetry, landmark motion, and clinical grading features.",
            style = MaterialTheme.typography.bodyMedium,
            color = AppColors.TextSecondary,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))
        ClinicalCard {
            StageChecklistRow("Geometry quality verified", StageState.DONE)
            Spacer(modifier = Modifier.height(14.dp))
            StageChecklistRow("Regional asymmetry compared", StageState.DONE)
            Spacer(modifier = Modifier.height(14.dp))
            StageChecklistRow("Preparing clinical summary", StageState.ACTIVE)
        }
    }
}

@Composable
private fun ErrorState(message: String) {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Card(
            shape = RoundedCornerShape(20.dp),
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
    val confidence = probabilities[grade] ?: 0.0

    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        EyebrowLabel("Assessment summary")
        Text("Completed today", style = MaterialTheme.typography.bodySmall, color = AppColors.TextSecondary)
    }
    Spacer(modifier = Modifier.height(16.dp))

    ClinicalCard {
        Text(
            "HOUSE-BRACKMANN GRADE",
            style = MaterialTheme.typography.labelMedium,
            color = AppColors.TextSecondary,
            modifier = Modifier.align(Alignment.CenterHorizontally)
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            "Grade $grade",
            style = MaterialTheme.typography.displayLarge,
            color = gradeColor,
            modifier = Modifier.align(Alignment.CenterHorizontally)
        )
        Text(
            gradeLabel + " dysfunction",
            style = MaterialTheme.typography.bodyMedium,
            color = AppColors.TextSecondary,
            modifier = Modifier.align(Alignment.CenterHorizontally)
        )
        Spacer(modifier = Modifier.height(18.dp))
        LinearProgressIndicator(
            progress = { confidence.toFloat() },
            modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
            color = AppColors.AccentAqua,
            trackColor = AppColors.SurfaceMuted
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "Assessment confidence: ${"%.0f".format(confidence * 100)}%" +
                if (framesUsed > 1) " · averaged across $framesUsed frames" else "",
            style = MaterialTheme.typography.bodySmall,
            color = AppColors.TextSecondary,
            modifier = Modifier.align(Alignment.CenterHorizontally)
        )
    }

    Spacer(modifier = Modifier.height(14.dp))

    ClinicalCard {
        Row {
            Box(
                modifier = Modifier
                    .size(30.dp)
                    .clip(RoundedCornerShape(9.dp))
                    .background(AppColors.PrimaryDark),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Filled.FactCheck, contentDescription = null, tint = AppColors.AccentAquaSoft, modifier = Modifier.size(16.dp))
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                Text("Clinical interpretation", style = MaterialTheme.typography.titleMedium, color = AppColors.Primary)
                Spacer(modifier = Modifier.height(4.dp))
                Text(explanation, style = MaterialTheme.typography.bodyMedium, color = AppColors.TextPrimary)
            }
        }
    }

    Spacer(modifier = Modifier.height(14.dp))

    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        SecondaryButton(
            text = "View details",
            onClick = onViewExplanation,
            modifier = Modifier.weight(1f)
        )
        Button(
            onClick = onViewDigitalTwin,
            modifier = Modifier.weight(1f).height(56.dp),
            shape = RoundedCornerShape(28.dp),
            colors = ButtonDefaults.buttonColors(containerColor = AppColors.Primary)
        ) {
            Icon(Icons.Filled.Timeline, contentDescription = null, tint = AppColors.TextOnPrimary, modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(8.dp))
            Text("Recovery trend", style = MaterialTheme.typography.labelLarge, color = AppColors.TextOnPrimary)
        }
    }

    Spacer(modifier = Modifier.height(18.dp))

    ConfidenceBars(probabilities = probabilities, selectedGrade = grade)

    Spacer(modifier = Modifier.height(14.dp))

    DisclaimerBanner()
}
