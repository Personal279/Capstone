package com.pes.facialparalysis.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Checkbox
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Shown EVERY time the user downloads or shares a report. The checkbox starts unchecked on every
 * showing (state is local to this composable), so the image is never included by default.
 * Unchecked = report is generated without any image.
 */
@Composable
fun ExportConsentDialog(
    confirmLabel: String,
    onConfirm: (includeImage: Boolean) -> Unit,
    onDismiss: () -> Unit
) {
    var includeImage by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Create report?") },
        text = {
            Column {
                Text(
                    "The report contains the grade, measurements and AI explanation. " +
                            "Once created, it can be saved or sent outside this app.",
                    fontSize = 14.sp
                )
                Spacer(modifier = Modifier.height(16.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .toggleable(
                            value = includeImage,
                            role = Role.Checkbox,
                            onValueChange = { includeImage = it }
                        )
                ) {
                    Checkbox(checked = includeImage, onCheckedChange = null)
                    Spacer(modifier = Modifier.width(10.dp))
                    Text("I agree to include my face images (first and latest) in this report", fontSize = 14.sp)
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    if (includeImage) "Images will be included in this report."
                    else "The report will be created without any image.",
                    fontSize = 12.sp
                )
            }
        },
        confirmButton = { TextButton(onClick = { onConfirm(includeImage) }) { Text(confirmLabel) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } }
    )
}