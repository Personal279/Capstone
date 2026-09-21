package com.pes.facialparalysis.ui.screens

import android.graphics.Color as AndroidColor
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.formatter.ValueFormatter
import com.pes.facialparalysis.data.AppDatabase
import com.pes.facialparalysis.data.AssessmentRecord
import com.pes.facialparalysis.data.DigitalTwinSnapshot
import com.pes.facialparalysis.data.SelectedPatientHolder
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.components.AiDisclosure
import com.pes.facialparalysis.ui.theme.components.DemoDataBadge
import com.pes.facialparalysis.ui.theme.components.DisclaimerBanner
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun DigitalTwinScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val patientId = SelectedPatientHolder.patientId

    Column(modifier = Modifier.fillMaxSize().background(AppColors.Background)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.Filled.ArrowBack, contentDescription = "Back", tint = AppColors.TextPrimary)
            }
            Spacer(modifier = Modifier.width(4.dp))
            Column {
                Text("Digital Twin", fontSize = 18.sp, fontWeight = FontWeight.Bold, color = AppColors.TextPrimary)
                Text(
                    SelectedPatientHolder.patientName ?: "",
                    fontSize = 12.sp,
                    color = AppColors.TextSecondary
                )
            }
        }

        if (patientId == null) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No patient selected.", color = AppColors.TextSecondary)
            }
            return
        }

        val records by remember(patientId) {
            AppDatabase.getDatabase(context).assessmentDao().getRecordsForPatientAscending(patientId)
        }.collectAsState(initial = null)

        if (records == null) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = AppColors.Primary)
            }
            return
        }

        val snapshot = DigitalTwinSnapshot.from(records!!, patientId)
        var showDemo by remember { mutableStateOf(false) }

        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp)
        ) {
            if (snapshot.latest == null) {
                EmptyTwinState()
            } else {
                CurrentGradeCard(snapshot.latest)
                Spacer(modifier = Modifier.height(14.dp))
                SymmetryMetricsCard(snapshot.latest)
                Spacer(modifier = Modifier.height(14.dp))
                ChangeSincePreviousCard(snapshot)
                Spacer(modifier = Modifier.height(14.dp))
                SectionHeader("Progress timeline")
                if (snapshot.history.size >= 2) {
                    TimelineChart(snapshot.history)
                    Spacer(modifier = Modifier.height(12.dp))
                } else {
                    InfoNote("Timeline needs at least two assessments for this patient. Only the baseline is available so far.")
                    Spacer(modifier = Modifier.height(12.dp))
                }
                SectionHeader("Previous assessments")
                snapshot.history.reversed().forEach { record ->
                    AssessmentRow(record)
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }

            Spacer(modifier = Modifier.height(20.dp))
            DemoExampleSection(
                expanded = showDemo,
                onToggle = { showDemo = !showDemo },
                patientId = patientId,
                seedGrade = snapshot.latest?.grade ?: 3
            )

            Spacer(modifier = Modifier.height(16.dp))
            DisclaimerBanner(
                text = AiDisclosure.FULL + " The Digital Twin shows observed model outputs and " +
                    "landmark-derived measurements over time — it does not predict future recovery."
            )
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
private fun EmptyTwinState() {
    Box(modifier = Modifier.fillMaxWidth().padding(vertical = 40.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("No assessments yet", fontSize = 15.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "Run an assessment for this patient to start their Digital Twin.",
                fontSize = 13.sp,
                color = AppColors.TextSecondary
            )
        }
    }
}

@Composable
private fun SectionHeader(text: String) {
    Text(
        text = text,
        fontSize = 13.sp,
        fontWeight = FontWeight.SemiBold,
        color = AppColors.TextPrimary,
        modifier = Modifier.padding(bottom = 8.dp)
    )
}

@Composable
private fun InfoNote(text: String) {
    Text(text, fontSize = 12.sp, color = AppColors.TextSecondary, lineHeight = 16.sp)
}

@Composable
private fun TwinCard(content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
        border = BorderStroke(1.dp, AppColors.Border),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp), content = content)
    }
}

@Composable
private fun CurrentGradeCard(latest: AssessmentRecord) {
    val gradeColor = AppColors.gradeColor(latest.grade)
    TwinCard {
        Text("Current AI-estimated grade", fontSize = 12.sp, color = AppColors.TextSecondary)
        Spacer(modifier = Modifier.height(6.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(44.dp).clip(RoundedCornerShape(12.dp)).background(gradeColor.copy(alpha = 0.12f)),
                contentAlignment = Alignment.Center
            ) {
                Text("G${latest.grade}", fontSize = 15.sp, fontWeight = FontWeight.Bold, color = gradeColor)
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                Text(latest.gradeLabel, fontSize = 16.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary)
                Text(
                    "Model confidence: ${"%.0f".format(latest.confidence * 100)}%",
                    fontSize = 12.sp,
                    color = AppColors.TextSecondary
                )
            }
        }
    }
}

@Composable
private fun SymmetryMetricsCard(latest: AssessmentRecord) {
    TwinCard {
        Text("Facial symmetry (from landmark measurements)", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = AppColors.TextPrimary)
        Spacer(modifier = Modifier.height(10.dp))
        SymmetryRow("Overall facial symmetry", latest.faiValue)
        SymmetryRow("Eye symmetry", latest.eyeAsymmetry)
        SymmetryRow("Mouth symmetry", latest.mouthAsymmetry)
        SymmetryRow("Eyebrow symmetry", latest.browAsymmetry)
        SymmetryRow("Cheek symmetry", latest.cheekAsymmetry)
        SymmetryRow("Jaw / lower face symmetry", latest.jawAsymmetry)
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            "Shown as measured asymmetry % between left and right side (lower = more symmetric).",
            fontSize = 11.sp,
            color = AppColors.TextSecondary
        )
    }
}

@Composable
private fun SymmetryRow(label: String, asymmetryPercent: Double) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, fontSize = 13.sp, color = AppColors.TextPrimary)
        Text("${"%.1f".format(asymmetryPercent)}%", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary)
    }
}

@Composable
private fun ChangeSincePreviousCard(snapshot: DigitalTwinSnapshot) {
    TwinCard {
        Text("Change since previous assessment", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = AppColors.TextPrimary)
        Spacer(modifier = Modifier.height(8.dp))
        val previous = snapshot.previous
        val latest = snapshot.latest
        if (previous == null || latest == null) {
            InfoNote("This is the first recorded assessment for this patient — nothing to compare yet.")
        } else {
            val gradeDelta = latest.grade - previous.grade
            val faiDelta = latest.faiValue - previous.faiValue
            val gradeText = when {
                gradeDelta == 0 -> "Grade unchanged (Grade ${latest.grade})"
                gradeDelta < 0 -> "Grade improved from ${previous.grade} to ${latest.grade}"
                else -> "Grade changed from ${previous.grade} to ${latest.grade}"
            }
            Text(gradeText, fontSize = 13.sp, color = AppColors.TextPrimary)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "Overall measured asymmetry ${if (faiDelta <= 0) "decreased" else "increased"} by ${"%.1f".format(kotlin.math.abs(faiDelta))} points",
                fontSize = 13.sp,
                color = AppColors.TextSecondary
            )
        }
    }
}

@Composable
private fun AssessmentRow(record: AssessmentRecord) {
    val dateStr = remember(record.timestamp) {
        SimpleDateFormat("MMM d, yyyy - h:mm a", Locale.getDefault()).format(Date(record.timestamp))
    }
    val gradeColor = AppColors.gradeColor(record.grade)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .border(BorderStroke(1.dp, AppColors.Border), RoundedCornerShape(10.dp))
            .background(AppColors.Surface)
            .padding(14.dp)
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text("Grade ${record.grade} - ${record.gradeLabel}", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary)
            Spacer(modifier = Modifier.height(2.dp))
            Text(dateStr, fontSize = 11.sp, color = AppColors.TextSecondary)
        }
        Box(
            modifier = Modifier.background(gradeColor.copy(alpha = 0.12f), RoundedCornerShape(6.dp)).padding(horizontal = 8.dp, vertical = 3.dp)
        ) {
            Text("${"%.0f".format(record.confidence * 100)}%", fontSize = 11.sp, fontWeight = FontWeight.Medium, color = gradeColor)
        }
    }
}

@Composable
private fun TimelineChart(history: List<AssessmentRecord>) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .border(BorderStroke(1.dp, AppColors.Border), RoundedCornerShape(12.dp))
            .background(AppColors.Surface)
            .padding(12.dp)
    ) {
        AndroidView(
            modifier = Modifier.fillMaxWidth().height(200.dp),
            factory = { ctx -> LineChart(ctx) },
            update = { chart ->
                val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())
                val entries = history.mapIndexed { index, record -> Entry(index.toFloat(), record.grade.toFloat()) }
                val dataSet = LineDataSet(entries, "Grade").apply {
                    color = AndroidColor.parseColor("#0F6E7A")
                    valueTextColor = AndroidColor.parseColor("#1B2B2E")
                    valueTextSize = 11f
                    lineWidth = 2.5f
                    circleRadius = 4f
                    setCircleColor(AndroidColor.parseColor("#0F6E7A"))
                    setDrawFilled(true)
                    fillColor = AndroidColor.parseColor("#0F6E7A")
                    fillAlpha = 25
                    setDrawValues(true)
                }
                chart.data = LineData(dataSet)
                chart.description.isEnabled = false
                chart.legend.isEnabled = false
                chart.setBackgroundColor(AndroidColor.TRANSPARENT)
                chart.axisRight.isEnabled = false
                chart.axisLeft.textColor = AndroidColor.parseColor("#5C6B6E")
                chart.axisLeft.axisMinimum = 0f
                chart.axisLeft.axisMaximum = 6f
                chart.axisLeft.setDrawGridLines(true)
                chart.axisLeft.gridColor = AndroidColor.parseColor("#DCE5E6")
                chart.xAxis.position = XAxis.XAxisPosition.BOTTOM
                chart.xAxis.textColor = AndroidColor.parseColor("#5C6B6E")
                chart.xAxis.setDrawGridLines(false)
                chart.xAxis.granularity = 1f
                chart.xAxis.valueFormatter = object : ValueFormatter() {
                    override fun getFormattedValue(value: Float): String {
                        val index = value.toInt()
                        return history.getOrNull(index)?.let { dateFormat.format(Date(it.timestamp)) } ?: ""
                    }
                }
                chart.setTouchEnabled(true)
                chart.notifyDataSetChanged()
                chart.invalidate()
            }
        )
    }
}

/** Collapsible, clearly-labeled synthetic timeline so the Digital Twin UI can be demoed even
 * when a patient doesn't yet have enough real assessments. Never touches the real data above. */
@Composable
private fun DemoExampleSection(expanded: Boolean, onToggle: () -> Unit, patientId: Int, seedGrade: Int) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .border(BorderStroke(1.dp, AppColors.Border), RoundedCornerShape(12.dp))
            .background(AppColors.SurfaceMuted)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().clickable(onClick = onToggle).padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Preview with example data", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary)
                Spacer(modifier = Modifier.width(8.dp))
                DemoDataBadge()
            }
            Icon(
                imageVector = if (expanded) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                contentDescription = null,
                tint = AppColors.TextSecondary
            )
        }
        if (expanded) {
            val demo = DigitalTwinSnapshot.demo(patientId, seedGrade)
            Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 4.dp).padding(bottom = 14.dp)) {
                Text(
                    "Illustrative timeline only — not real data for this patient.",
                    fontSize = 11.sp,
                    color = AppColors.Warning
                )
                Spacer(modifier = Modifier.height(10.dp))
                TimelineChart(demo.history)
                Spacer(modifier = Modifier.height(10.dp))
                demo.history.reversed().forEach { record ->
                    AssessmentRow(record)
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}
