package com.pes.facialparalysis.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pes.facialparalysis.data.XaiDataHolder
import com.pes.facialparalysis.ml.FaiCalculator
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.components.AiDisclosure
import com.pes.facialparalysis.ui.theme.components.ClinicalBackButton
import com.pes.facialparalysis.ui.theme.components.ClinicalCard
import com.pes.facialparalysis.ui.theme.components.ClinicalDivider
import com.pes.facialparalysis.ui.theme.components.ClinicalMetricRow
import com.pes.facialparalysis.ui.theme.components.ConfidenceBars
import com.pes.facialparalysis.ui.theme.components.DisclaimerBanner
import com.pes.facialparalysis.ui.theme.components.EyebrowLabel
import com.pes.facialparalysis.ui.theme.components.LandmarkAsymmetryOverlay
import com.pes.facialparalysis.ui.theme.components.LimeImportanceOverlay
import com.pes.facialparalysis.ui.theme.components.NotADiagnosisBanner
import com.pes.facialparalysis.ui.theme.components.PossibleScanIssuesCard
import com.pes.facialparalysis.ui.theme.components.RegionImportanceBars
import com.pes.facialparalysis.ui.theme.components.ScreenHeading
import com.pes.facialparalysis.ui.theme.components.clinicalBackgroundBrush

private val regionDisplayName = mapOf(
    "eye" to "Eye region",
    "mouth" to "Mouth region",
    "brow" to "Eyebrow region",
    "cheek" to "Cheek region",
    "jaw" to "Lower face / jaw region"
)

@Composable
fun XaiExplanationScreen(onBack: () -> Unit) {
    val bitmap = XaiDataHolder.bitmap
    val landmarks = XaiDataHolder.landmarks
    val regions = XaiDataHolder.regions
    val grade = XaiDataHolder.grade
    val probabilities = XaiDataHolder.probabilities
    val summary = XaiDataHolder.summarySentence
    val bullets = XaiDataHolder.bulletLabels
    val limeResult = XaiDataHolder.limeResult
    val scanQuality = XaiDataHolder.scanQuality

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(clinicalBackgroundBrush())
    ) {
        Spacer(modifier = Modifier.windowInsetsTopHeight(WindowInsets.statusBars))
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ClinicalBackButton(onClick = onBack)
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                EyebrowLabel("Explainable assessment")
                Text("Why this prediction?", style = MaterialTheme.typography.headlineMedium, color = AppColors.TextPrimary)
            }
        }

        if (bitmap == null || grade == null) {
            Box(modifier = Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
                Text(
                    "No explanation data available for this session. Run a new assessment to view its AI explanation.",
                    fontSize = 14.sp,
                    color = AppColors.TextSecondary
                )
            }
            return
        }

        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
        ) {
            NotADiagnosisBanner()
            Spacer(modifier = Modifier.height(16.dp))

            if (limeResult != null) {
                SectionLabel("Why this grade? Important facial regions")
                ClinicalCard(contentPadding = PaddingValues(0.dp)) {
                    Box(modifier = Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = null,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(20.dp))
                        )
                        if (landmarks != null) {
                            LimeImportanceOverlay(
                                landmarks = landmarks,
                                importance = limeResult.regionImportance,
                                modifier = Modifier.fillMaxSize()
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Highlight strength shows how much each region's visibility pushed the " +
                        "AI toward its predicted grade (via LIME, tested on this exact image). It does " +
                        "not mean these areas are medically affected.",
                    fontSize = 11.sp,
                    color = AppColors.TextSecondary,
                    lineHeight = 15.sp
                )
                Spacer(modifier = Modifier.height(16.dp))
                RegionImportanceBars(limeResult.regionImportance)
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Local explanation fit: ${"%.0f".format(limeResult.localModelFitQuality * 100)}% " +
                        "— how well this simplified explanation tracks the AI model's actual behavior on this image.",
                    fontSize = 11.sp,
                    color = AppColors.TextSecondary
                )
                Spacer(modifier = Modifier.height(20.dp))
            }

            SectionLabel("Detected landmarks & measured asymmetry overlay")
            ClinicalCard(contentPadding = PaddingValues(0.dp)) {
                Box(modifier = Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(20.dp))
                    )
                    if (landmarks != null) {
                        LandmarkAsymmetryOverlay(
                            landmarks = landmarks,
                            regions = regions,
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Dots mark facial landmarks used for measurement. Colored markers scale with " +
                    "each region's measured asymmetry — this is a geometric visualization of real landmark " +
                    "data, not a model-internal saliency map (the on-device model does not expose one).",
                fontSize = 11.sp,
                color = AppColors.TextSecondary,
                lineHeight = 15.sp
            )

            Spacer(modifier = Modifier.height(20.dp))
            SectionLabel("Measured asymmetry by region")
            RegionMeasurementsCard(regions)

            Spacer(modifier = Modifier.height(20.dp))
            SectionLabel("How confident was the AI?")
            ConfidenceBars(probabilities = probabilities, selectedGrade = grade, title = "Confidence by grade")

            if (scanQuality != null) {
                Spacer(modifier = Modifier.height(20.dp))
                SectionLabel("Could this result be incorrect?")
                PossibleScanIssuesCard(
                    detectedFactors = scanQuality.detectedFactors,
                    unverifiedFactors = scanQuality.unverifiedPossibleFactors
                )
            }

            Spacer(modifier = Modifier.height(20.dp))
            SectionLabel("Limitations")
            ClinicalCard {
                Text(
                    text = "• This is a single automated estimate, not a clinical exam.\n" +
                        "• The model was trained on a limited dataset and may not generalize to every " +
                        "face, lighting condition, or camera.\n" +
                        "• The explanation above describes the model's own behavior on this image — it " +
                        "is not a measurement of nerve or muscle function.\n" +
                        "• Only a qualified healthcare professional can diagnose facial paralysis.",
                    fontSize = 12.sp,
                    color = AppColors.TextPrimary,
                    lineHeight = 17.sp
                )
            }

            Spacer(modifier = Modifier.height(20.dp))
            SectionLabel("AI explanation — why this grade?")
            ClinicalCard(containerColor = AppColors.PrimaryLight) {
                if (limeResult != null) {
                    Text(
                        text = limeResult.summarySentence,
                        fontSize = 13.sp,
                        color = AppColors.TextPrimary,
                        fontWeight = FontWeight.Medium,
                        lineHeight = 18.sp
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                }
                Text(
                    "Landmark measurement notes",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = AppColors.TextSecondary
                )
                Spacer(modifier = Modifier.height(4.dp))
                bullets.forEach { label ->
                    Text("●  $label", fontSize = 13.sp, color = AppColors.TextPrimary, fontWeight = FontWeight.Medium)
                    Spacer(modifier = Modifier.height(4.dp))
                }
                Spacer(modifier = Modifier.height(6.dp))
                Text(summary, fontSize = 13.sp, color = AppColors.TextPrimary, lineHeight = 18.sp)
            }

            Spacer(modifier = Modifier.height(16.dp))
            DisclaimerBanner(text = AiDisclosure.FULL + " These regions contributed to the model's prediction; " +
                "they do not confirm a diagnosis.")
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
private fun SectionLabel(text: String) {
    EyebrowLabel(text, color = AppColors.TextSecondary, modifier = Modifier.padding(bottom = 8.dp))
}

@Composable
private fun RegionMeasurementsCard(regions: List<FaiCalculator.RegionAsymmetry>) {
    ClinicalCard {
        val sorted = regions.sortedByDescending { it.asymmetryPercent }
        sorted.forEachIndexed { index, region ->
            ClinicalMetricRow(
                label = regionDisplayName[region.region] ?: region.region,
                value = "${"%.1f".format(region.asymmetryPercent)}%",
                valueColor = if (region.asymmetryPercent > 15.0) AppColors.AccentRose else AppColors.AccentAmber
            )
            if (index != sorted.lastIndex) {
                Spacer(modifier = Modifier.height(8.dp))
                ClinicalDivider()
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            "Asymmetry % = difference between the left- and right-side landmark measurement for that region.",
            fontSize = 11.sp,
            color = AppColors.TextSecondary
        )
    }
}
