package com.pes.facialparalysis

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pes.facialparalysis.data.SelectedPatientHolder
import com.pes.facialparalysis.navigation.AppNavHost
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.FacialParalysisTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FacialParalysisTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Box(modifier = Modifier.padding(innerPadding)) {
                        AppNavHost()
                    }
                }
            }
        }
    }
}

@Composable
fun HomeScreen(
    onUploadClick: () -> Unit = {},
    onRecordClick: () -> Unit = {},
    onCameraClick: () -> Unit = {},
    onHistoryClick: () -> Unit = {},
    onSwitchPatientClick: () -> Unit = {}
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppColors.Background)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(AppColors.Surface)
                .border(width = 1.dp, color = AppColors.Border)
                .padding(horizontal = 20.dp, vertical = 14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "PATIENT",
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Medium,
                    color = AppColors.TextSecondary,
                    letterSpacing = 1.sp
                )
                Text(
                    text = SelectedPatientHolder.patientName ?: "None selected",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = AppColors.TextPrimary
                )
            }
            TextButton(onClick = onSwitchPatientClick) {
                Text("Switch", color = AppColors.Primary, fontSize = 13.sp, fontWeight = FontWeight.Medium)
            }
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(28.dp))

            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(AppColors.PrimaryLight),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Face,
                    contentDescription = null,
                    tint = AppColors.Primary,
                    modifier = Modifier.size(30.dp)
                )
            }

            Spacer(modifier = Modifier.height(14.dp))

            Text(
                text = "NeuroVision AI",
                fontSize = 21.sp,
                fontWeight = FontWeight.Bold,
                color = AppColors.TextPrimary
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = "Facial Paralysis Detection & Grading",
                fontSize = 13.sp,
                color = AppColors.TextSecondary
            )

            Spacer(modifier = Modifier.height(24.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
                border = androidx.compose.foundation.BorderStroke(1.dp, AppColors.Border),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
            ) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.Shield,
                            contentDescription = null,
                            tint = AppColors.Primary,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "Data Privacy",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = AppColors.TextPrimary
                        )
                    }
                    Spacer(modifier = Modifier.height(12.dp))
                    PrivacyRow("Images never leave your device")
                    PrivacyRow("Fully offline, on-device analysis")
                    PrivacyRow("Temporary files deleted after processing")
                }
            }

            Spacer(modifier = Modifier.height(28.dp))

            Text(
                text = "NEW ASSESSMENT",
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium,
                color = AppColors.TextSecondary,
                letterSpacing = 1.sp,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(10.dp))

            ActionButton(text = "Open Camera", icon = Icons.Filled.CameraAlt, filled = true, onClick = onCameraClick)
            Spacer(modifier = Modifier.height(10.dp))
            ActionButton(text = "Upload Image", icon = Icons.Filled.Upload, filled = false, onClick = onUploadClick)
            Spacer(modifier = Modifier.height(10.dp))
            ActionButton(text = "Record Video", icon = Icons.Filled.Videocam, filled = false, onClick = onRecordClick)
            Spacer(modifier = Modifier.height(10.dp))
            ActionButton(text = "View History", icon = Icons.Filled.History, filled = false, onClick = onHistoryClick)

            Spacer(modifier = Modifier.weight(1f))

            Text(
                text = "Powered by on-device AI",
                fontSize = 11.sp,
                color = AppColors.TextSecondary,
                modifier = Modifier.padding(bottom = 20.dp)
            )
        }
    }
}

@Composable
private fun PrivacyRow(text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 3.dp)
    ) {
        Icon(
            imageVector = Icons.Filled.CheckCircle,
            contentDescription = null,
            tint = AppColors.Primary,
            modifier = Modifier.size(14.dp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(text = text, fontSize = 13.sp, color = AppColors.TextSecondary)
    }
}

@Composable
private fun ActionButton(
    text: String,
    icon: ImageVector,
    filled: Boolean,
    onClick: () -> Unit
) {
    if (filled) {
        Button(
            onClick = onClick,
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(10.dp),
            colors = ButtonDefaults.buttonColors(containerColor = AppColors.Primary)
        ) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp), tint = AppColors.TextOnPrimary)
            Spacer(modifier = Modifier.width(8.dp))
            Text(text, fontSize = 15.sp, fontWeight = FontWeight.Medium, color = AppColors.TextOnPrimary)
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(10.dp),
            border = androidx.compose.foundation.BorderStroke(1.dp, AppColors.Border),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.TextPrimary)
        ) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp), tint = AppColors.Primary)
            Spacer(modifier = Modifier.width(8.dp))
            Text(text, fontSize = 15.sp, fontWeight = FontWeight.Medium)
        }
    }
}

@Preview(showBackground = true)
@Composable
fun HomeScreenPreview() {
    FacialParalysisTheme {
        HomeScreen()
    }
}