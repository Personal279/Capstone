package com.pes.facialparalysis.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
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
import com.pes.facialparalysis.ui.theme.components.ConfidenceBars
import com.pes.facialparalysis.ui.theme.components.DisclaimerBanner
import com.pes.facialparalysis.ui.theme.components.LandmarkAsymmetryOverlay

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

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppColors.Background)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.Filled.ArrowBack, contentDescription = "Back", tint = AppColors.TextPrimary)
            }
            Spacer(modifier = Modifier.width(4.dp))
            Column {
                Text("Why this prediction?", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = AppColors.TextPrimary)
                Text("AI explanation for this assessment", fontSize = 12.sp, color = AppColors.TextSecondary)
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
            SectionLabel("1–3. Image, detected landmarks & asymmetry-region overlay")
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
                border = BorderStroke(1.dp, AppColors.Border),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
            ) {
                Box(modifier = Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(12.dp))
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
            SectionLabel("4. Measured asymmetry by region")
            RegionMeasurementsCard(regions)

            Spacer(modifier = Modifier.height(20.dp))
            SectionLabel("5. Model prediction and confidence")
            ConfidenceBars(probabilities = probabilities, selectedGrade = grade, title = "Confidence by grade")

            Spacer(modifier = Modifier.height(20.dp))
            SectionLabel("6. Plain-language explanation")
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = AppColors.PrimaryLight),
                border = BorderStroke(1.dp, AppColors.Border),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    bullets.forEach { label ->
                        Text("●  $label", fontSize = 13.sp, color = AppColors.TextPrimary, fontWeight = FontWeight.Medium)
                        Spacer(modifier = Modifier.height(4.dp))
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(summary, fontSize = 13.sp, color = AppColors.TextPrimary, lineHeight = 18.sp)
                }
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
    Text(
        text = text.uppercase(),
        fontSize = 11.sp,
        fontWeight = FontWeight.SemiBold,
        letterSpacing = 0.5.sp,
        color = AppColors.TextSecondary,
        modifier = Modifier.padding(bottom = 8.dp)
    )
}

@Composable
private fun RegionMeasurementsCard(regions: List<FaiCalculator.RegionAsymmetry>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
        border = BorderStroke(1.dp, AppColors.Border),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            regions.sortedByDescending { it.asymmetryPercent }.forEach { region ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = regionDisplayName[region.region] ?: region.region,
                        fontSize = 13.sp,
                        color = AppColors.TextPrimary
                    )
                    Text(
                        text = "${"%.1f".format(region.asymmetryPercent)}%",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium,
                        color = AppColors.TextPrimary
                    )
                }
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                "Asymmetry % = difference between the left- and right-side landmark measurement for that region.",
                fontSize = 11.sp,
                color = AppColors.TextSecondary
            )
        }
    }
}
