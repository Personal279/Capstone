package com.pes.facialparalysis.ui.screens

import android.graphics.Bitmap
import android.widget.Toast
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pes.facialparalysis.data.AppDatabase
import com.pes.facialparalysis.data.AssessmentRecord
import com.pes.facialparalysis.report.ReportData
import com.pes.facialparalysis.report.ReportExporter
import com.pes.facialparalysis.report.decodeSampledBitmap
import com.pes.facialparalysis.report.parseDoubleMap
import com.pes.facialparalysis.report.parseProbabilities
import com.pes.facialparalysis.report.regionDisplayNames
import com.pes.facialparalysis.report.relativeShares
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.components.ClinicalBackButton
import com.pes.facialparalysis.ui.theme.components.ClinicalCard
import com.pes.facialparalysis.ui.theme.components.ClinicalDivider
import com.pes.facialparalysis.ui.theme.components.ClinicalMetricRow
import com.pes.facialparalysis.ui.theme.components.ConfidenceBars
import com.pes.facialparalysis.ui.theme.components.DisclaimerBanner
import com.pes.facialparalysis.ui.theme.components.EyebrowLabel
import com.pes.facialparalysis.ui.theme.components.PrimaryButton
import com.pes.facialparalysis.ui.theme.components.SecondaryButton
import com.pes.facialparalysis.ui.theme.components.clinicalBackgroundBrush
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private enum class ExportAction { DOWNLOAD, SHARE }

private class LoadedRecord(
    val data: ReportData?,
    val initial: Bitmap?,
    val latest: Bitmap?
)

/**
 * One screen that shows the whole saved report for an assessment (result, regions, AI explanation,
 * first + latest image) loaded purely from the database, plus download/share as PDF.
 */
@Composable
fun RecordDetailScreen(recordId: Int, onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var loading by remember { mutableStateOf(true) }
    var loaded by remember { mutableStateOf<LoadedRecord?>(null) }
    var pending by remember { mutableStateOf<ExportAction?>(null) }
    var busy by remember { mutableStateOf(false) }

    LaunchedEffect(recordId) {
        loaded = withContext(Dispatchers.IO) {
            val db = AppDatabase.getDatabase(context)
            val rec = db.assessmentDao().getById(recordId)
            if (rec == null) {
                LoadedRecord(null, null, null)
            } else {
                val patient = db.patientDao().getById(rec.patientId)
                LoadedRecord(
                    data = ReportData(
                        patientName = patient?.name ?: "Patient",
                        record = rec,
                        initialImagePath = patient?.initialImagePath,
                        latestImagePath = patient?.latestImagePath
                    ),
                    initial = decodeSampledBitmap(patient?.initialImagePath, 900),
                    latest = decodeSampledBitmap(patient?.latestImagePath, 900)
                )
            }
        }
        loading = false
    }

    Column(modifier = Modifier.fillMaxSize().background(clinicalBackgroundBrush())) {
        Spacer(modifier = Modifier.windowInsetsTopHeight(WindowInsets.statusBars))
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ClinicalBackButton(onClick = onBack)
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                EyebrowLabel("Saved assessment")
                Text("Full report", style = MaterialTheme.typography.headlineMedium, color = AppColors.TextPrimary)
            }
        }

        val data = loaded?.data
        when {
            loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = AppColors.Primary)
            }
            data == null -> Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
                Text("This assessment could not be found.", color = AppColors.TextSecondary)
            }
            else -> {
                val r = data.record
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .verticalScroll(rememberScrollState())
                        .padding(horizontal = 20.dp)
                ) {
                    GradeCard(r, data.patientName)
                    Spacer(Modifier.height(16.dp))

                    if (loaded?.initial != null || loaded?.latest != null) {
                        EyebrowLabel("Images", color = AppColors.TextSecondary, modifier = Modifier.padding(bottom = 8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            ImageTile("First test (initial)", loaded?.initial, Modifier.weight(1f))
                            ImageTile("Most recent test", loaded?.latest, Modifier.weight(1f))
                        }
                        Spacer(Modifier.height(16.dp))
                    }

                    EyebrowLabel("Measured asymmetry by region", color = AppColors.TextSecondary, modifier = Modifier.padding(bottom = 8.dp))
                    ClinicalCard {
                        val rows = listOf(
                            "eye" to r.eyeAsymmetry, "mouth" to r.mouthAsymmetry, "brow" to r.browAsymmetry,
                            "cheek" to r.cheekAsymmetry, "jaw" to r.jawAsymmetry
                        ).sortedByDescending { it.second }
                        rows.forEachIndexed { i, (key, v) ->
                            ClinicalMetricRow(
                                label = regionDisplayNames[key] ?: key,
                                value = "%.1f%%".format(v),
                                valueColor = if (v > 15.0) AppColors.AccentRose else AppColors.AccentAmber
                            )
                            if (i != rows.lastIndex) {
                                Spacer(Modifier.height(8.dp)); ClinicalDivider(); Spacer(Modifier.height(8.dp))
                            }
                        }
                    }

                    val probs = parseProbabilities(r.probabilitiesJson)
                    if (probs.isNotEmpty()) {
                        Spacer(Modifier.height(16.dp))
                        ConfidenceBars(probabilities = probs, selectedGrade = r.grade, title = "Confidence by grade")
                    }

                    Spacer(Modifier.height(16.dp))
                    EyebrowLabel("AI explanation - why this grade?", color = AppColors.TextSecondary, modifier = Modifier.padding(bottom = 8.dp))
                    ClinicalCard(containerColor = AppColors.PrimaryLight) {
                        if (r.limeSummary.isNotBlank()) {
                            Text(r.limeSummary, fontSize = 13.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary, lineHeight = 18.sp)
                            Spacer(Modifier.height(10.dp))
                        }
                        val shares = relativeShares(parseDoubleMap(r.limeImportanceJson))
                        shares.entries.sortedByDescending { it.value }.forEach { (k, v) ->
                            ShareBar(regionDisplayNames[k] ?: k, v)
                            Spacer(Modifier.height(8.dp))
                        }
                        r.topRegionsCsv.split(",").filter { it.isNotBlank() }.forEach {
                            Text("●  ${it.trim()}", fontSize = 13.sp, color = AppColors.TextPrimary)
                            Spacer(Modifier.height(4.dp))
                        }
                        if (r.xaiExplanation.isNotBlank()) {
                            Spacer(Modifier.height(6.dp))
                            Text(r.xaiExplanation, fontSize = 13.sp, color = AppColors.TextPrimary, lineHeight = 18.sp)
                        }
                    }

                    Spacer(Modifier.height(16.dp))
                    DisclaimerBanner()
                    Spacer(Modifier.height(16.dp))

                    if (busy) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp, color = AppColors.Primary)
                            Spacer(Modifier.width(10.dp))
                            Text("Creating report...", fontSize = 13.sp, color = AppColors.TextSecondary)
                        }
                        Spacer(Modifier.height(12.dp))
                    }
                    SecondaryButton(
                        text = "Download PDF",
                        onClick = { if (!busy) pending = ExportAction.DOWNLOAD },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(Modifier.height(10.dp))
                    PrimaryButton(
                        text = "Share PDF",
                        onClick = { if (!busy) pending = ExportAction.SHARE },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(Modifier.height(24.dp))
                    Spacer(Modifier.windowInsetsBottomHeight(WindowInsets.navigationBars))
                }
            }
        }
    }

    // Consent is asked on EVERY export: the dialog is recreated each time, checkbox starts unchecked.
    val action = pending
    val reportData = loaded?.data
    if (action != null && reportData != null) {
        ExportConsentDialog(
            confirmLabel = if (action == ExportAction.SHARE) "Create & share" else "Create & download",
            onDismiss = { pending = null },
            onConfirm = { includeImage ->
                pending = null
                busy = true
                scope.launch {
                    try {
                        val file = ReportExporter.createPdf(context, reportData, includeImage)
                        when (action) {
                            ExportAction.SHARE -> ReportExporter.share(context, file)
                            ExportAction.DOWNLOAD -> {
                                val ok = ReportExporter.saveToDownloads(context, file)
                                Toast.makeText(
                                    context,
                                    if (ok) "Saved to Downloads" else "Couldn't save directly - use Share PDF and choose Save",
                                    Toast.LENGTH_LONG
                                ).show()
                            }
                        }
                    } catch (e: Exception) {
                        Toast.makeText(context, "Couldn't create the report. Please try again.", Toast.LENGTH_LONG).show()
                    } finally {
                        busy = false
                    }
                }
            }
        )
    }
}

@Composable
private fun GradeCard(r: AssessmentRecord, patientName: String) {
    val dateStr = SimpleDateFormat("MMM d, yyyy - h:mm a", Locale.getDefault()).format(Date(r.timestamp))
    ClinicalCard {
        Text(patientName, style = MaterialTheme.typography.bodySmall, color = AppColors.TextSecondary,
            modifier = Modifier.align(Alignment.CenterHorizontally))
        Text("HOUSE-BRACKMANN GRADE", style = MaterialTheme.typography.labelMedium, color = AppColors.TextSecondary,
            modifier = Modifier.align(Alignment.CenterHorizontally))
        Spacer(Modifier.height(6.dp))
        Text("Grade ${r.grade}", style = MaterialTheme.typography.displayLarge, color = AppColors.gradeColor(r.grade),
            modifier = Modifier.align(Alignment.CenterHorizontally))
        Text("${r.gradeLabel} dysfunction", style = MaterialTheme.typography.bodyMedium, color = AppColors.TextSecondary,
            modifier = Modifier.align(Alignment.CenterHorizontally))
        Spacer(Modifier.height(12.dp))
        Text("Confidence ${"%.0f".format(r.confidence * 100)}% · $dateStr", style = MaterialTheme.typography.bodySmall,
            color = AppColors.TextSecondary, modifier = Modifier.align(Alignment.CenterHorizontally))
    }
}

@Composable
private fun ImageTile(label: String, bitmap: Bitmap?, modifier: Modifier) {
    Column(modifier) {
        Text(label, fontSize = 11.sp, color = AppColors.TextSecondary)
        Spacer(Modifier.height(4.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(3f / 4f)
                .clip(RoundedCornerShape(14.dp))
                .background(AppColors.SurfaceMuted),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(bitmap.asImageBitmap(), contentDescription = label, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
            } else {
                Text("No image", fontSize = 12.sp, color = AppColors.TextSecondary)
            }
        }
    }
}

@Composable
private fun ShareBar(label: String, percent: Double) {
    Column {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, fontSize = 12.sp, color = AppColors.TextPrimary)
            Text("%.0f%%".format(percent), fontSize = 12.sp, color = AppColors.TextSecondary)
        }
        Spacer(Modifier.height(3.dp))
        LinearProgressIndicator(
            progress = { (percent / 100.0).toFloat().coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth().height(5.dp).clip(RoundedCornerShape(3.dp)),
            color = AppColors.AccentAqua,
            trackColor = AppColors.SurfaceMuted
        )
    }
}