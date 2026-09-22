package com.pes.facialparalysis.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.components.ClinicalCard
import com.pes.facialparalysis.ui.theme.components.EyebrowLabel
import com.pes.facialparalysis.ui.theme.components.PrimaryButton
import com.pes.facialparalysis.ui.theme.components.clinicalBackgroundBrush

private data class Benefit(val icon: ImageVector, val text: String)

private val benefits = listOf(
    Benefit(Icons.Filled.CheckCircle, "Structured, repeatable capture guidance"),
    Benefit(Icons.Filled.Timeline, "Clear symmetry and movement trends"),
    Benefit(Icons.Filled.Psychology, "Explainable visual AI findings")
)

/**
 * Clinical-companion primer shown before a guided capture. Introduces the flow the user is
 * about to go through; it carries no capture/ML logic of its own.
 */
@Composable
fun OnboardingScreen(onStartGuidedCapture: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(clinicalBackgroundBrush())
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 24.dp)
    ) {
        Spacer(modifier = Modifier.windowInsetsTopHeight(WindowInsets.statusBars))
        Spacer(modifier = Modifier.height(20.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            EyebrowLabel("Clinical companion")
            Text("01/06", style = MaterialTheme.typography.bodySmall, color = AppColors.TextSecondary)
        }
        Spacer(modifier = Modifier.height(18.dp))

        ClinicalCard {
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(AppColors.PrimaryDark),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Filled.Face, contentDescription = null, tint = AppColors.AccentAquaSoft)
            }
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Assessment with context.",
                style = MaterialTheme.typography.displayMedium,
                color = AppColors.TextPrimary
            )
            Spacer(modifier = Modifier.height(10.dp))
            Text(
                text = "A guided visual assessment helps you document facial symmetry and movement between clinical visits.",
                style = MaterialTheme.typography.bodyMedium,
                color = AppColors.TextSecondary
            )
        }

        Spacer(modifier = Modifier.height(20.dp))

        benefits.forEach { benefit ->
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(benefit.icon, contentDescription = null, tint = AppColors.Primary, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Text(benefit.text, style = MaterialTheme.typography.bodyMedium, color = AppColors.TextPrimary)
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        Text(
            text = "This visual tool supports, but does not replace, clinical care.",
            style = MaterialTheme.typography.bodySmall,
            color = AppColors.TextSecondary,
            modifier = Modifier.padding(vertical = 14.dp)
        )

        PrimaryButton(
            text = "Start guided capture",
            onClick = onStartGuidedCapture,
            showArrow = true,
            modifier = Modifier.padding(bottom = 20.dp)
        )
        Spacer(modifier = Modifier.windowInsetsBottomHeight(WindowInsets.navigationBars))
    }
}
