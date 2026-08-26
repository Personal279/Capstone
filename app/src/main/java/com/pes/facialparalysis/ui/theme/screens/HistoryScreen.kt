package com.pes.facialparalysis.ui.screens

import android.graphics.Color as AndroidColor
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ChevronRight
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
import com.pes.facialparalysis.data.SelectedPatientHolder
import com.pes.facialparalysis.ui.theme.AppColors
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun HistoryScreen(
    onRecordClick: (AssessmentRecord) -> Unit = {},
    onStartNewAssessment: () -> Unit = {}
) {
    val context = LocalContext.current
    val patientId = SelectedPatientHolder.patientId

    if (patientId == null) {
        Box(
            modifier = Modifier.fillMaxSize().background(AppColors.Background),
            contentAlignment = Alignment.Center
        ) {
            Text("No patient selected.", color = AppColors.TextSecondary)
        }
        return
    }

    // Distinguish "still loading" from "genuinely empty" so a slow DB read
    // doesn't briefly flash a misleading empty state.
    var isLoading by remember(patientId) { mutableStateOf(true) }
    val records by remember(patientId) {
        AppDatabase.getDatabase(context).assessmentDao().getRecordsForPatient(patientId)
    }.collectAsState(initial = emptyList())

    LaunchedEffect(records, patientId) {
        isLoading = false
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppColors.Background)
            .padding(20.dp)
    ) {
        Text(
            text = "Progression history",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            color = AppColors.TextPrimary
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = SelectedPatientHolder.patientName ?: "",
            fontSize = 13.sp,
            color = AppColors.TextSecondary
        )
        Spacer(modifier = Modifier.height(16.dp))

        when {
            isLoading -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = AppColors.Primary, strokeWidth = 2.5.dp)
                }
            }
            records.isEmpty() -> {
                EmptyHistoryState(onStartNewAssessment)
            }
            else -> {
                if (records.size > 1) {
                    ProgressionChart(records.sortedBy { it.timestamp })
                    Spacer(modifier = Modifier.height(16.dp))
                    GradeLegend()
                    Spacer(modifier = Modifier.height(20.dp))
                }

                Text(
                    text = "Past assessments",
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = AppColors.TextPrimary
                )
                Spacer(modifier = Modifier.height(8.dp))

                LazyColumn {
                    items(records, key = { it.id }) { record ->
                        AssessmentCard(record, onClick = { onRecordClick(record) })
                        Spacer(modifier = Modifier.height(10.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun EmptyHistoryState(onStartNewAssessment: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "No assessments yet",
            fontSize = 15.sp,
            fontWeight = FontWeight.Medium,
            color = AppColors.TextPrimary
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "Assessments you run for this patient will show up here.",
            fontSize = 13.sp,
            color = AppColors.TextSecondary
        )
        Spacer(modifier = Modifier.height(20.dp))
        Button(
            onClick = onStartNewAssessment,
            colors = ButtonDefaults.buttonColors(containerColor = AppColors.Primary),
            shape = RoundedCornerShape(10.dp)
        ) {
            Icon(Icons.Filled.Add, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text("New assessment", color = AppColors.TextOnPrimary)
        }
    }
}

@Composable
private fun GradeLegend() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        listOf(1, 2, 3, 4, 5, 6).forEach { grade ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(CircleShape)
                        .background(AppColors.gradeColor(grade))
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(text = "$grade", fontSize = 11.sp, color = AppColors.TextSecondary)
            }
        }
    }
}

@Composable
private fun ProgressionChart(records: List<AssessmentRecord>) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .border(BorderStroke(1.dp, AppColors.Border), RoundedCornerShape(12.dp))
            .background(AppColors.Surface)
            .padding(12.dp)
    ) {
        AndroidView(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp),
            factory = { ctx -> LineChart(ctx) },
            update = { chart ->
                val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())
                val entries = records.mapIndexed { index, record ->
                    Entry(index.toFloat(), record.grade.toFloat())
                }
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

                // X-axis now labels actual assessment dates instead of bare
                // indices — the entire reason to show a trend chart is
                // "when did this change," so the axis needs to say when.
                chart.xAxis.position = XAxis.XAxisPosition.BOTTOM
                chart.xAxis.textColor = AndroidColor.parseColor("#5C6B6E")
                chart.xAxis.setDrawGridLines(false)
                chart.xAxis.granularity = 1f
                chart.xAxis.valueFormatter = object : ValueFormatter() {
                    override fun getFormattedValue(value: Float): String {
                        val index = value.toInt()
                        return records.getOrNull(index)?.let { dateFormat.format(Date(it.timestamp)) } ?: ""
                    }
                }

                chart.setTouchEnabled(true)
                chart.notifyDataSetChanged()
                chart.invalidate()
            }
        )
    }
}

@Composable
private fun AssessmentCard(record: AssessmentRecord, onClick: () -> Unit) {
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
            .clickable(onClick = onClick)
            .padding(16.dp)
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = "Grade ${record.grade} - ${record.gradeLabel}",
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                color = AppColors.TextPrimary
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(text = dateStr, fontSize = 12.sp, color = AppColors.TextSecondary)
        }
        Box(
            modifier = Modifier
                .background(gradeColor.copy(alpha = 0.12f), RoundedCornerShape(6.dp))
                .padding(horizontal = 10.dp, vertical = 4.dp)
        ) {
            Text(
                text = "${"%.0f".format(record.confidence * 100)}%",
                fontSize = 12.sp,
                fontWeight = FontWeight.Medium,
                color = gradeColor
            )
        }
        Spacer(modifier = Modifier.width(8.dp))
        Icon(
            imageVector = Icons.Filled.ChevronRight,
            contentDescription = null,
            tint = AppColors.TextSecondary,
            modifier = Modifier.size(18.dp)
        )
    }
}