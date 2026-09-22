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
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Notifications
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
import com.pes.facialparalysis.data.DoctorNote
import com.pes.facialparalysis.data.Reminder
import com.pes.facialparalysis.data.SelectedPatientHolder
import com.pes.facialparalysis.ml.TrajectoryEstimator
import com.pes.facialparalysis.reminders.ReminderScheduler
import com.pes.facialparalysis.ui.theme.AppColors
import com.pes.facialparalysis.ui.theme.components.AiDisclosure
import com.pes.facialparalysis.ui.theme.components.ClinicalBackButton
import com.pes.facialparalysis.ui.theme.components.ClinicalCard
import com.pes.facialparalysis.ui.theme.components.DemoDataBadge
import com.pes.facialparalysis.ui.theme.components.DisclaimerBanner
import com.pes.facialparalysis.ui.theme.components.EyebrowLabel
import com.pes.facialparalysis.ui.theme.components.ScreenHeading
import com.pes.facialparalysis.ui.theme.components.clinicalBackgroundBrush
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun DigitalTwinScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val patientId = SelectedPatientHolder.patientId

    Column(modifier = Modifier.fillMaxSize().background(clinicalBackgroundBrush())) {
        Spacer(modifier = Modifier.windowInsetsTopHeight(WindowInsets.statusBars))
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            ClinicalBackButton(onClick = onBack)
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                EyebrowLabel("Recovery monitoring")
                Text(
                    SelectedPatientHolder.patientName ?: "Digital Twin",
                    fontSize = 12.sp,
                    color = AppColors.TextSecondary
                )
            }
        }
        Text(
            "Your recovery trend.",
            style = MaterialTheme.typography.displayMedium,
            color = AppColors.TextPrimary,
            modifier = Modifier.padding(horizontal = 20.dp)
        )
        Spacer(modifier = Modifier.height(12.dp))

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
                SymmetryAndGradeSummaryRow(snapshot)
                Spacer(modifier = Modifier.height(14.dp))
                CurrentGradeCard(snapshot.latest)
                Spacer(modifier = Modifier.height(14.dp))
                SymmetryMetricsCard(snapshot.latest)
                Spacer(modifier = Modifier.height(14.dp))
                ChangeSincePreviousCard(snapshot)
                Spacer(modifier = Modifier.height(14.dp))
                SectionHeader("Condition graph")
                if (snapshot.history.size >= 2) {
                    ConditionGraphSection(snapshot.history)
                    Spacer(modifier = Modifier.height(12.dp))
                } else {
                    InfoNote("A trend graph needs at least two assessments for this patient. Only the baseline is available so far.")
                    Spacer(modifier = Modifier.height(12.dp))
                }
                SectionHeader("Recovery trajectory")
                RecoveryTrajectoryCard(snapshot)
                Spacer(modifier = Modifier.height(14.dp))
                SectionHeader("Predicted trajectory")
                PredictedTrajectoryCard(snapshot.history)
                Spacer(modifier = Modifier.height(14.dp))
                SectionHeader("Recovery information — Grade ${snapshot.latest.grade}")
                GradeRecoveryInfoCard(snapshot.latest.grade)
                Spacer(modifier = Modifier.height(14.dp))
                SectionHeader("Digital twin timeline — previous assessments")
                snapshot.history.reversed().forEach { record ->
                    AssessmentRow(record)
                    Spacer(modifier = Modifier.height(8.dp))
                }
                Spacer(modifier = Modifier.height(14.dp))
                SectionHeader("Doctor's notes")
                DoctorNotesSection(patientId)
                Spacer(modifier = Modifier.height(14.dp))
                SectionHeader("Reminders")
                RemindersSection(patientId)
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
    EyebrowLabel(text, color = AppColors.TextPrimary, modifier = Modifier.padding(bottom = 8.dp))
}

@Composable
private fun InfoNote(text: String) {
    Text(text, fontSize = 12.sp, color = AppColors.TextSecondary, lineHeight = 16.sp)
}

@Composable
private fun TwinCard(content: @Composable ColumnScope.() -> Unit) {
    ClinicalCard(content = content)
}

/**
 * "Current symmetry" / "assessment grade" summary pair, mirroring the recovery-dashboard design
 * reference. Symmetry% here is simply (100 - measured overall asymmetry%) from the same real
 * [AssessmentRecord.faiValue] used everywhere else on this screen — not a separate metric.
 */
@Composable
private fun SymmetryAndGradeSummaryRow(snapshot: DigitalTwinSnapshot) {
    val latest = snapshot.latest ?: return
    val currentSymmetry = (100.0 - latest.faiValue).coerceIn(0.0, 100.0)
    val baselineSymmetry = snapshot.baseline?.let { (100.0 - it.faiValue).coerceIn(0.0, 100.0) }
    val deltaText = baselineSymmetry?.let {
        val delta = currentSymmetry - it
        "${if (delta >= 0) "+" else ""}${"%.0f".format(delta)}% from baseline"
    } ?: "Baseline assessment"
    val gradeTrend = when {
        snapshot.previous == null -> "First recorded assessment"
        latest.grade < snapshot.previous!!.grade -> "Improving pattern"
        latest.grade > snapshot.previous!!.grade -> "Worsening pattern"
        else -> "Stable pattern"
    }

    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        ClinicalCard(modifier = Modifier.weight(1f)) {
            Text("CURRENT SYMMETRY", style = MaterialTheme.typography.labelMedium, color = AppColors.TextSecondary)
            Spacer(modifier = Modifier.height(6.dp))
            Text("${"%.0f".format(currentSymmetry)}%", style = MaterialTheme.typography.headlineLarge, color = AppColors.AccentAqua)
            Spacer(modifier = Modifier.height(2.dp))
            Text(deltaText, style = MaterialTheme.typography.bodySmall, color = AppColors.TextSecondary)
        }
        ClinicalCard(modifier = Modifier.weight(1f)) {
            Text("ASSESSMENT GRADE", style = MaterialTheme.typography.labelMedium, color = AppColors.TextSecondary)
            Spacer(modifier = Modifier.height(6.dp))
            Text("Grade ${latest.grade}", style = MaterialTheme.typography.headlineLarge, color = AppColors.gradeColor(latest.grade))
            Spacer(modifier = Modifier.height(2.dp))
            Text(gradeTrend, style = MaterialTheme.typography.bodySmall, color = AppColors.TextSecondary)
        }
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

private enum class GraphMetric(val label: String) {
    OVERALL("Overall"), GRADE("Grade"), EYE("Eye"), MOUTH("Mouth"), BROW("Brow"), CHEEK("Cheek"), JAW("Jaw")
}

@Composable
private fun ConditionGraphSection(history: List<AssessmentRecord>) {
    var metric by remember { mutableStateOf(GraphMetric.OVERALL) }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .border(BorderStroke(1.dp, AppColors.Border), RoundedCornerShape(12.dp))
            .background(AppColors.Surface)
            .padding(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            GraphMetric.entries.forEach { m ->
                val selected = m == metric
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (selected) AppColors.Primary else AppColors.SurfaceMuted)
                        .clickable { metric = m }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Text(
                        m.label,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium,
                        color = if (selected) AppColors.TextOnPrimary else AppColors.TextSecondary
                    )
                }
            }
        }
        val values = history.map { record ->
            when (metric) {
                GraphMetric.OVERALL -> record.faiValue
                GraphMetric.GRADE -> record.grade.toDouble()
                GraphMetric.EYE -> record.eyeAsymmetry
                GraphMetric.MOUTH -> record.mouthAsymmetry
                GraphMetric.BROW -> record.browAsymmetry
                GraphMetric.CHEEK -> record.cheekAsymmetry
                GraphMetric.JAW -> record.jawAsymmetry
            }
        }
        val axisMax = if (metric == GraphMetric.GRADE) 6f else null
        MetricLineChart(history, values, axisMax)
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = when (metric) {
                GraphMetric.GRADE -> "Y-axis: AI-estimated grade (1–5) per assessment date."
                GraphMetric.OVERALL -> "Y-axis: overall measured facial asymmetry % per assessment date."
                else -> "Y-axis: measured ${metric.label.lowercase()} region asymmetry % per assessment date."
            },
            fontSize = 11.sp,
            color = AppColors.TextSecondary
        )
    }
}

@Composable
private fun MetricLineChart(history: List<AssessmentRecord>, values: List<Double>, axisMax: Float?) {
    AndroidView(
        modifier = Modifier.fillMaxWidth().height(200.dp),
        factory = { ctx -> LineChart(ctx) },
        update = { chart ->
            val dateFormat = SimpleDateFormat("MMM d", Locale.getDefault())
            val entries = values.mapIndexed { index, v -> Entry(index.toFloat(), v.toFloat()) }
            val dataSet = LineDataSet(entries, "Value").apply {
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
            if (axisMax != null) chart.axisLeft.axisMaximum = axisMax else chart.axisLeft.resetAxisMaximum()
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

/** Historical overall-asymmetry points (solid) plus a dashed segment out to the [TrajectoryEstimator]'s
 * real 14-day linear projection — never a fabricated point, only what the regression actually output. */
@Composable
private fun PredictedTrajectoryChart(history: List<AssessmentRecord>, result: TrajectoryEstimator.TrajectoryResult) {
    val projected = result.projectedFaiIn14Days ?: return
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
                val lastIndex = history.size - 1

                val historicalEntries = history.mapIndexed { index, record -> Entry(index.toFloat(), record.faiValue.toFloat()) }
                val historicalSet = LineDataSet(historicalEntries, "Observed").apply {
                    color = AndroidColor.parseColor("#0F6E7A")
                    valueTextColor = AndroidColor.parseColor("#1B2B2E")
                    valueTextSize = 10f
                    lineWidth = 2.5f
                    circleRadius = 4f
                    setCircleColor(AndroidColor.parseColor("#0F6E7A"))
                    setDrawFilled(false)
                    setDrawValues(false)
                }

                // Dashed projection segment from the last real point to the projected point.
                val projectionEntries = listOf(
                    Entry(lastIndex.toFloat(), history.last().faiValue.toFloat()),
                    Entry((lastIndex + 1).toFloat(), projected.toFloat())
                )
                val projectionSet = LineDataSet(projectionEntries, "Projected (+14d)").apply {
                    color = AndroidColor.parseColor("#B5651D")
                    valueTextColor = AndroidColor.parseColor("#B5651D")
                    valueTextSize = 10f
                    lineWidth = 2.5f
                    circleRadius = 4f
                    setCircleColor(AndroidColor.parseColor("#B5651D"))
                    enableDashedLine(14f, 8f, 0f)
                    setDrawValues(true)
                }

                chart.data = LineData(historicalSet, projectionSet)
                chart.description.isEnabled = false
                chart.legend.isEnabled = true
                chart.legend.textColor = AndroidColor.parseColor("#5C6B6E")
                chart.legend.textSize = 10f
                chart.setBackgroundColor(AndroidColor.TRANSPARENT)
                chart.axisRight.isEnabled = false
                chart.axisLeft.textColor = AndroidColor.parseColor("#5C6B6E")
                chart.axisLeft.axisMinimum = 0f
                chart.axisLeft.resetAxisMaximum()
                chart.axisLeft.setDrawGridLines(true)
                chart.axisLeft.gridColor = AndroidColor.parseColor("#DCE5E6")
                chart.xAxis.position = XAxis.XAxisPosition.BOTTOM
                chart.xAxis.textColor = AndroidColor.parseColor("#5C6B6E")
                chart.xAxis.setDrawGridLines(false)
                chart.xAxis.granularity = 1f
                chart.xAxis.valueFormatter = object : ValueFormatter() {
                    override fun getFormattedValue(value: Float): String {
                        val index = value.toInt()
                        if (index == lastIndex + 1) return "+14d"
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

@Composable
private fun RecoveryTrajectoryCard(snapshot: DigitalTwinSnapshot) {
    TwinCard {
        val baseline = snapshot.baseline
        val latest = snapshot.latest
        if (baseline == null || latest == null) {
            InfoNote("Not enough assessments yet to describe a trend.")
            return@TwinCard
        }
        val change = latest.faiValue - baseline.faiValue
        val trendDirection = when {
            kotlin.math.abs(change) < 1.0 -> "Little change observed"
            change < 0 -> "Decreasing measured asymmetry"
            else -> "Increasing measured asymmetry"
        }
        Text("Observed assessment trend", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = AppColors.TextPrimary)
        if (snapshot.history.size >= 2) {
            Spacer(modifier = Modifier.height(8.dp))
            MetricLineChart(snapshot.history, snapshot.history.map { it.faiValue }, axisMax = null)
        }
        Spacer(modifier = Modifier.height(8.dp))
        SymmetryRow("Baseline overall asymmetry", baseline.faiValue)
        SymmetryRow("Current overall asymmetry", latest.faiValue)
        Spacer(modifier = Modifier.height(4.dp))
        Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
            Text("Change in AI-estimated facial asymmetry", fontSize = 13.sp, color = AppColors.TextPrimary)
            Text("${if (change <= 0) "-" else "+"}${"%.1f".format(kotlin.math.abs(change))}%", fontSize = 13.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary)
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text("Assessments recorded: ${snapshot.history.size}", fontSize = 12.sp, color = AppColors.TextSecondary)
        Text(trendDirection, fontSize = 12.sp, color = AppColors.TextSecondary)
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            "This reflects observed model outputs over time, not guaranteed clinical recovery.",
            fontSize = 11.sp,
            color = AppColors.TextSecondary
        )
    }
}

@Composable
private fun PredictedTrajectoryCard(history: List<AssessmentRecord>) {
    val result = remember(history) { TrajectoryEstimator.estimate(history) }
    TwinCard {
        if (!result.hasEnoughData) {
            Text(
                "Additional longitudinal assessments are required to estimate a trajectory.",
                fontSize = 13.sp,
                color = AppColors.TextSecondary
            )
        } else {
            Text("Experimental model-estimated trajectory", fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = AppColors.TextPrimary)
            Spacer(modifier = Modifier.height(8.dp))
            PredictedTrajectoryChart(history.sortedBy { it.timestamp }, result)
            Spacer(modifier = Modifier.height(8.dp))
            Text(result.trendDescription ?: "", fontSize = 13.sp, color = AppColors.TextPrimary)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "Projected overall asymmetry in ~14 days: ${"%.1f".format(result.projectedFaiIn14Days ?: 0.0)}%",
                fontSize = 13.sp,
                color = AppColors.TextPrimary
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "Fit of this trend to past data: ${"%.0f".format((result.fitQuality ?: 0.0) * 100)}%",
                fontSize = 12.sp,
                color = AppColors.TextSecondary
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                "Not a medical prognosis. This is a simple statistical projection from this patient's " +
                    "own past assessments and can be wrong, especially with few data points.",
                fontSize = 11.sp,
                color = AppColors.TextSecondary
            )
        }
    }
}

private val gradeRecoveryInfo = mapOf(
    1 to "No significant asymmetry was estimated. Continue routine monitoring and mention any new " +
        "facial changes to a healthcare professional.",
    2 to "Mild asymmetry was estimated. General monitoring is often recommended; a healthcare " +
        "professional can advise on any next steps appropriate for you.",
    3 to "Moderate asymmetry was estimated. Many patients at this level are followed by a healthcare " +
        "professional for monitoring and guidance; specific care plans vary by individual.",
    4 to "Moderately severe asymmetry was estimated. Ongoing professional evaluation is commonly " +
        "advised at this level; only a qualified clinician can determine an appropriate plan.",
    5 to "Severe asymmetry was estimated. Prompt evaluation by a qualified healthcare professional is " +
        "commonly advised; this app cannot determine treatment for you."
)

@Composable
private fun GradeRecoveryInfoCard(grade: Int) {
    TwinCard {
        Text(
            gradeRecoveryInfo[grade] ?: "Educational information is not available for this grade.",
            fontSize = 13.sp,
            color = AppColors.TextPrimary,
            lineHeight = 18.sp
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "General educational information only. Not a prescription, treatment plan, or diagnosis.",
            fontSize = 11.sp,
            color = AppColors.TextSecondary
        )
    }
}

@Composable
private fun DoctorNotesSection(patientId: Int) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val notes by remember(patientId) {
        AppDatabase.getDatabase(context).doctorNoteDao().getNotesForPatient(patientId)
    }.collectAsState(initial = emptyList())
    var showAddDialog by remember { mutableStateOf(false) }

    Column {
        if (notes.isEmpty()) {
            InfoNote("No doctor's notes yet for this patient.")
            Spacer(modifier = Modifier.height(8.dp))
        } else {
            notes.forEach { note ->
                DoctorNoteRow(note)
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
        OutlinedButton(
            onClick = { showAddDialog = true },
            modifier = Modifier.fillMaxWidth().height(44.dp),
            shape = RoundedCornerShape(10.dp),
            border = BorderStroke(1.dp, AppColors.Border),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.Primary)
        ) {
            Icon(Icons.Filled.Add, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text("Add doctor's note", fontSize = 13.sp, fontWeight = FontWeight.Medium)
        }
    }

    if (showAddDialog) {
        AddNoteDialog(
            onDismiss = { showAddDialog = false },
            onConfirm = { text, category, reminder ->
                scope.launch {
                    val db = AppDatabase.getDatabase(context)
                    var reminderId: Int? = null
                    if (reminder != null) {
                        val ownedReminder = reminder.copy(patientId = patientId)
                        val id = db.reminderDao().insert(ownedReminder).toInt()
                        reminderId = id
                        ReminderScheduler.schedule(context, ownedReminder.copy(id = id))
                    }
                    db.doctorNoteDao().insert(
                        DoctorNote(
                            patientId = patientId,
                            text = text,
                            createdAt = System.currentTimeMillis(),
                            category = category,
                            linkedReminderId = reminderId
                        )
                    )
                    showAddDialog = false
                }
            }
        )
    }
}

@Composable
private fun DoctorNoteRow(note: DoctorNote) {
    val dateStr = remember(note.createdAt) {
        SimpleDateFormat("MMM d, yyyy", Locale.getDefault()).format(Date(note.createdAt))
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .border(BorderStroke(1.dp, AppColors.Border), RoundedCornerShape(10.dp))
            .background(AppColors.Surface)
            .padding(14.dp)
    ) {
        Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
            Text(note.category, fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = AppColors.Primary)
            Text(dateStr, fontSize = 11.sp, color = AppColors.TextSecondary)
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(note.text, fontSize = 13.sp, color = AppColors.TextPrimary, lineHeight = 18.sp)
        if (note.linkedReminderId != null) {
            Spacer(modifier = Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.Notifications, contentDescription = null, tint = AppColors.TextSecondary, modifier = Modifier.size(12.dp))
                Spacer(modifier = Modifier.width(4.dp))
                Text("Reminder scheduled", fontSize = 11.sp, color = AppColors.TextSecondary)
            }
        }
    }
}

@Composable
private fun AddNoteDialog(
    onDismiss: () -> Unit,
    onConfirm: (text: String, category: String, reminder: Reminder?) -> Unit
) {
    var text by remember { mutableStateOf("") }
    var category by remember { mutableStateOf("General") }
    var addReminder by remember { mutableStateOf(false) }
    var hour by remember { mutableStateOf(18) }
    var minute by remember { mutableStateOf(0) }
    var selectedDays by remember { mutableStateOf(setOf<Int>()) }
    val categories = listOf("General", "Physiotherapy", "Medication", "Follow-up")
    val dayOptions = listOf(
        Calendar.MONDAY to "Mon", Calendar.TUESDAY to "Tue", Calendar.WEDNESDAY to "Wed",
        Calendar.THURSDAY to "Thu", Calendar.FRIDAY to "Fri", Calendar.SATURDAY to "Sat", Calendar.SUNDAY to "Sun"
    )

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New doctor's note", color = AppColors.TextPrimary) },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
                    categories.forEach { c ->
                        val selected = c == category
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(8.dp))
                                .background(if (selected) AppColors.Primary else AppColors.SurfaceMuted)
                                .clickable { category = c }
                                .padding(horizontal = 8.dp, vertical = 5.dp)
                        ) {
                            Text(c, fontSize = 10.sp, color = if (selected) AppColors.TextOnPrimary else AppColors.TextSecondary)
                        }
                    }
                }
                Spacer(modifier = Modifier.height(10.dp))
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    label = { Text("Note (e.g. \"Follow-up appointment on 30 September.\")") },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(10.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(checked = addReminder, onCheckedChange = { addReminder = it })
                    Text("Add reminder", fontSize = 13.sp, color = AppColors.TextPrimary)
                }
                if (addReminder) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Text("Days (leave all unchecked for a one-time reminder today):", fontSize = 11.sp, color = AppColors.TextSecondary)
                    Spacer(modifier = Modifier.height(4.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        dayOptions.forEach { (dayConst, label) ->
                            val selected = selectedDays.contains(dayConst)
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(6.dp))
                                    .background(if (selected) AppColors.Primary else AppColors.SurfaceMuted)
                                    .clickable {
                                        selectedDays = if (selected) selectedDays - dayConst else selectedDays + dayConst
                                    }
                                    .padding(horizontal = 6.dp, vertical = 4.dp)
                            ) {
                                Text(label, fontSize = 10.sp, color = if (selected) AppColors.TextOnPrimary else AppColors.TextSecondary)
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        OutlinedTextField(
                            value = hour.toString(),
                            onValueChange = { v -> v.toIntOrNull()?.let { if (it in 0..23) hour = it } },
                            label = { Text("Hour (0-23)") },
                            modifier = Modifier.weight(1f)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        OutlinedTextField(
                            value = minute.toString(),
                            onValueChange = { v -> v.toIntOrNull()?.let { if (it in 0..59) minute = it } },
                            label = { Text("Minute") },
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    val reminder = if (addReminder) {
                        Reminder(
                            patientId = 0, // overwritten by caller's actual patientId path via copy below
                            title = category,
                            notes = text,
                            hour = hour,
                            minute = minute,
                            daysOfWeekCsv = selectedDays.joinToString(","),
                            oneTimeDateMillis = if (selectedDays.isEmpty()) System.currentTimeMillis() else 0L,
                            createdAt = System.currentTimeMillis()
                        )
                    } else null
                    if (text.isNotBlank()) onConfirm(text.trim(), category, reminder)
                },
                enabled = text.isNotBlank()
            ) { Text("Save", color = AppColors.Primary) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel", color = AppColors.TextSecondary) } }
    )
}

@Composable
private fun RemindersSection(patientId: Int) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val reminders by remember(patientId) {
        AppDatabase.getDatabase(context).reminderDao().getRemindersForPatient(patientId)
    }.collectAsState(initial = emptyList())
    var showAddDialog by remember { mutableStateOf(false) }

    Column {
        if (reminders.isEmpty()) {
            InfoNote("No active reminders for this patient. Reminders are only created from an explicit doctor's note or manual entry — never decided by the AI.")
            Spacer(modifier = Modifier.height(8.dp))
        } else {
            reminders.forEach { reminder ->
                ReminderRow(reminder, onCancel = {
                    scope.launch {
                        AppDatabase.getDatabase(context).reminderDao().deactivate(reminder.id)
                        ReminderScheduler.cancel(context, reminder.id)
                    }
                })
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
        OutlinedButton(
            onClick = { showAddDialog = true },
            modifier = Modifier.fillMaxWidth().height(44.dp),
            shape = RoundedCornerShape(10.dp),
            border = BorderStroke(1.dp, AppColors.Border),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = AppColors.Primary)
        ) {
            Icon(Icons.Filled.Add, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text("Add reminder", fontSize = 13.sp, fontWeight = FontWeight.Medium)
        }
    }

    if (showAddDialog) {
        AddReminderDialog(
            onDismiss = { showAddDialog = false },
            onConfirm = { reminder ->
                scope.launch {
                    val db = AppDatabase.getDatabase(context)
                    val ownedReminder = reminder.copy(patientId = patientId)
                    val id = db.reminderDao().insert(ownedReminder).toInt()
                    ReminderScheduler.schedule(context, ownedReminder.copy(id = id))
                    showAddDialog = false
                }
            }
        )
    }
}

@Composable
private fun AddReminderDialog(onDismiss: () -> Unit, onConfirm: (Reminder) -> Unit) {
    var title by remember { mutableStateOf("") }
    var notes by remember { mutableStateOf("") }
    var hour by remember { mutableStateOf(9) }
    var minute by remember { mutableStateOf(0) }
    var selectedDays by remember { mutableStateOf(setOf<Int>()) }
    val dayOptions = listOf(
        Calendar.MONDAY to "Mon", Calendar.TUESDAY to "Tue", Calendar.WEDNESDAY to "Wed",
        Calendar.THURSDAY to "Thu", Calendar.FRIDAY to "Fri", Calendar.SATURDAY to "Sat", Calendar.SUNDAY to "Sun"
    )

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("New reminder", color = AppColors.TextPrimary) },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                OutlinedTextField(
                    value = title,
                    onValueChange = { title = it },
                    label = { Text("Title (e.g. \"Physiotherapy\")") },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value = notes,
                    onValueChange = { notes = it },
                    label = { Text("Details (optional)") },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(10.dp))
                Text("Days (leave all unchecked for a one-time reminder today):", fontSize = 11.sp, color = AppColors.TextSecondary)
                Spacer(modifier = Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    dayOptions.forEach { (dayConst, label) ->
                        val selected = selectedDays.contains(dayConst)
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(6.dp))
                                .background(if (selected) AppColors.Primary else AppColors.SurfaceMuted)
                                .clickable {
                                    selectedDays = if (selected) selectedDays - dayConst else selectedDays + dayConst
                                }
                                .padding(horizontal = 6.dp, vertical = 4.dp)
                        ) {
                            Text(label, fontSize = 10.sp, color = if (selected) AppColors.TextOnPrimary else AppColors.TextSecondary)
                        }
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    OutlinedTextField(
                        value = hour.toString(),
                        onValueChange = { v -> v.toIntOrNull()?.let { if (it in 0..23) hour = it } },
                        label = { Text("Hour (0-23)") },
                        modifier = Modifier.weight(1f)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    OutlinedTextField(
                        value = minute.toString(),
                        onValueChange = { v -> v.toIntOrNull()?.let { if (it in 0..59) minute = it } },
                        label = { Text("Minute") },
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    if (title.isNotBlank()) {
                        onConfirm(
                            Reminder(
                                patientId = 0,
                                title = title.trim(),
                                notes = notes.trim(),
                                hour = hour,
                                minute = minute,
                                daysOfWeekCsv = selectedDays.joinToString(","),
                                oneTimeDateMillis = if (selectedDays.isEmpty()) System.currentTimeMillis() else 0L,
                                createdAt = System.currentTimeMillis()
                            )
                        )
                    }
                },
                enabled = title.isNotBlank()
            ) { Text("Save", color = AppColors.Primary) }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel", color = AppColors.TextSecondary) } }
    )
}

@Composable
private fun ReminderRow(reminder: Reminder, onCancel: () -> Unit) {
    val timeStr = "%02d:%02d".format(reminder.hour, reminder.minute)
    val scheduleStr = if (reminder.isRecurring) {
        val names = mapOf(
            Calendar.MONDAY to "Mon", Calendar.TUESDAY to "Tue", Calendar.WEDNESDAY to "Wed",
            Calendar.THURSDAY to "Thu", Calendar.FRIDAY to "Fri", Calendar.SATURDAY to "Sat", Calendar.SUNDAY to "Sun"
        )
        reminder.daysOfWeek().mapNotNull { names[it] }.joinToString(" / ") + " · $timeStr"
    } else {
        val dateStr = SimpleDateFormat("MMM d", Locale.getDefault()).format(Date(reminder.oneTimeDateMillis))
        "$dateStr · $timeStr"
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .border(BorderStroke(1.dp, AppColors.Border), RoundedCornerShape(10.dp))
            .background(AppColors.Surface)
            .padding(14.dp)
    ) {
        Icon(Icons.Filled.Notifications, contentDescription = null, tint = AppColors.Primary, modifier = Modifier.size(18.dp))
        Spacer(modifier = Modifier.width(10.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(reminder.title, fontSize = 13.sp, fontWeight = FontWeight.Medium, color = AppColors.TextPrimary)
            Spacer(modifier = Modifier.height(2.dp))
            Text(scheduleStr, fontSize = 12.sp, color = AppColors.TextSecondary)
        }
        TextButton(onClick = onCancel) { Text("Cancel", fontSize = 12.sp, color = AppColors.TextSecondary) }
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
