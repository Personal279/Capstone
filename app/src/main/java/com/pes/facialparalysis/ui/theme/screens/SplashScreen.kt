package com.pes.facialparalysis.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.components.EyebrowLabel
import com.pes.facialparalysis.ui.theme.components.FaceMeshVisualization
import com.pes.facialparalysis.ui.theme.components.PrimaryButton
import com.pes.facialparalysis.ui.theme.components.clinicalBackgroundBrush

/**
 * Branding entry screen shown once per app launch. Purely decorative — the face mesh here is
 * generic illustration, not a real scan, and no ML/capture logic runs on this screen.
 */
@Composable
fun SplashScreen(onBeginAssessment: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(clinicalBackgroundBrush())
            .padding(horizontal = 28.dp)
    ) {
        Spacer(modifier = Modifier.windowInsetsTopHeight(WindowInsets.statusBars))
        Spacer(modifier = Modifier.height(24.dp))
        EyebrowLabel("Facial Recovery AI")

        FaceMeshVisualization(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(vertical = 16.dp)
        )

        Column(modifier = Modifier.padding(bottom = 12.dp)) {
            Text(
                text = "A clearer view of recovery.",
                style = MaterialTheme.typography.displayLarge,
                color = AppColors.TextPrimary
            )
            Spacer(modifier = Modifier.height(14.dp))
            Text(
                text = "Clinical-grade facial movement insights, designed for calm, informed follow-up.",
                style = MaterialTheme.typography.bodyLarge,
                color = AppColors.TextSecondary
            )
        }

        PrimaryButton(
            text = "Begin assessment",
            onClick = onBeginAssessment,
            modifier = Modifier.padding(bottom = 20.dp)
        )
        Spacer(modifier = Modifier.windowInsetsBottomHeight(WindowInsets.navigationBars))
    }
}
