package com.pes.facialparalysis.ui.components

import android.content.Intent
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.pes.facialparalysis.data.ImageStore
import com.pes.facialparalysis.reporting.PdfReportGenerator
import com.pes.facialparalysis.reporting.ReportData
import com.pes.facialparalysis.ui.theme.AppColors
import java.io.File

/**
 * Drop this on ResultScreen/HistoryScreen. Every tap: asks consent -> builds the
 * PDF (with or without the photo per the answer) -> opens the system share sheet.
 *
 * `patientId` picks which stored image (latest_<patientId>.jpg) gets offered.
 * Requires a FileProvider declared in AndroidManifest.xml, authority
 * "<applicationId>.fileprovider", pointing at a paths.xml that includes
 * <cache-path name="reports" path="reports/"/>.
 */
@Composable
fun ExportReportButton(
    patientId: Int,
    patientName: String,
    data: ReportData,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    var showConsent by remember { mutableStateOf(false) }

    Button(
        onClick = { showConsent = true },
        modifier = modifier.height(52.dp),
        colors = ButtonDefaults.buttonColors(containerColor = AppColors.Primary)
    ) {
        Text("Download as PDF report")
    }

    if (showConsent) {
        ConsentDialog(
            onDismiss = { showConsent = false },
            onResult = { includeImage ->
                showConsent = false
                val imageFile: File? = if (includeImage) {
                    ImageStore.getLatest(context, patientId)
                } else {
                    null
                }
                val pdf = PdfReportGenerator.generate(context, data, imageFile)
                val uri = PdfReportGenerator.uriFor(context, pdf)

                val shareIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "application/pdf"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                context.startActivity(Intent.createChooser(shareIntent, "Share report"))
            }
        )
    }
}