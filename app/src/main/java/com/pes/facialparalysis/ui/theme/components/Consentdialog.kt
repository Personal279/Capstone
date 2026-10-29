package com.pes.facialparalysis.ui.components

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import com.pes.facialparalysis.ui.theme.AppColors

/**
 * Asked EVERY time before an image leaves the device (PDF export, share, upload —
 * whatever "sending" means in your flow). No "remember my choice" — team asked for
 * permission every time.
 *
 * onResult(true)  = include the photo
 * onResult(false) = proceed without the photo (report/export still happens)
 */
@Composable
fun ConsentDialog(
    onResult: (includeImage: Boolean) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Include photo?") },
        text = {
            Text(
                "This report can include the captured photo. You can send it without " +
                        "the photo instead — only the grade and analysis will be included."
            )
        },
        confirmButton = {
            TextButton(onClick = { onResult(true) }) {
                Text("Include photo", color = AppColors.Primary)
            }
        },
        dismissButton = {
            TextButton(onClick = { onResult(false) }) {
                Text("Without photo")
            }
        }
    )
}