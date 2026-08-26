package com.pes.facialparalysis.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pes.facialparalysis.data.AppDatabase
import com.pes.facialparalysis.data.Patient
import com.pes.facialparalysis.data.SelectedPatientHolder
import com.pes.facialparalysis.ui.theme.AppColors
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun PatientSelectionScreen(onPatientSelected: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val patients by remember {
        AppDatabase.getDatabase(context).patientDao().getAllPatients()
    }.collectAsState(initial = emptyList())

    var showAddDialog by remember { mutableStateOf(false) }
    var searchQuery by remember { mutableStateOf("") }

    val filteredPatients = remember(patients, searchQuery) {
        if (searchQuery.isBlank()) patients
        else patients.filter { it.name.contains(searchQuery.trim(), ignoreCase = true) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(AppColors.Background)
            .padding(24.dp)
    ) {
        Spacer(modifier = Modifier.height(20.dp))
        Text(
            text = "Select patient",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = AppColors.TextPrimary
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "Choose an existing profile or add a new one",
            fontSize = 13.sp,
            color = AppColors.TextSecondary
        )
        Spacer(modifier = Modifier.height(16.dp))

        if (patients.isNotEmpty()) {
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                placeholder = { Text("Search patients") },
                leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
                singleLine = true,
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(16.dp))
        }

        if (patients.isEmpty()) {
            Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
                Text("No patients yet. Add one to get started.", color = AppColors.TextSecondary)
            }
        } else if (filteredPatients.isEmpty()) {
            Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
                Text("No patients match \"$searchQuery\".", color = AppColors.TextSecondary, textAlign = TextAlign.Center)
            }
        } else {
            LazyColumn(modifier = Modifier.weight(1f)) {
                items(filteredPatients, key = { it.id }) { patient ->
                    PatientCard(patient) {
                        SelectedPatientHolder.patientId = patient.id
                        SelectedPatientHolder.patientName = patient.name
                        onPatientSelected()
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }
        }

        Button(
            onClick = { showAddDialog = true },
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(10.dp),
            colors = ButtonDefaults.buttonColors(containerColor = AppColors.Primary)
        ) {
            Text("Add new patient", fontSize = 15.sp, fontWeight = FontWeight.Medium, color = AppColors.TextOnPrimary)
        }
    }

    if (showAddDialog) {
        AddPatientDialog(
            existingNames = patients.map { it.name },
            onDismiss = { showAddDialog = false },
            onConfirm = { name, onDone ->
                scope.launch {
                    AppDatabase.getDatabase(context).patientDao().insert(
                        Patient(name = name, createdAt = System.currentTimeMillis())
                    )
                    onDone()
                    showAddDialog = false
                }
            }
        )
    }
}

@Composable
private fun AddPatientDialog(
    existingNames: List<String>,
    onDismiss: () -> Unit,
    onConfirm: (name: String, onDone: () -> Unit) -> Unit
) {
    var name by remember { mutableStateOf("") }
    var isSaving by remember { mutableStateOf(false) }

    val trimmed = name.trim()
    val isDuplicate = existingNames.any { it.equals(trimmed, ignoreCase = true) }
    val isValid = trimmed.isNotEmpty() && !isDuplicate

    fun submit() {
        if (!isValid || isSaving) return
        isSaving = true
        onConfirm(trimmed) { isSaving = false }
    }

    AlertDialog(
        onDismissRequest = { if (!isSaving) onDismiss() },
        title = { Text("New patient", color = AppColors.TextPrimary) },
        text = {
            Column {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("Patient name") },
                    singleLine = true,
                    isError = isDuplicate,
                    enabled = !isSaving,
                    keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(imeAction = ImeAction.Done),
                    keyboardActions = androidx.compose.foundation.text.KeyboardActions(onDone = { submit() })
                )
                if (isDuplicate) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "A patient with this name already exists.",
                        color = MaterialTheme.colorScheme.error,
                        fontSize = 12.sp
                    )
                }
            }
        },
        confirmButton = {
            TextButton(onClick = { submit() }, enabled = isValid && !isSaving) {
                if (isSaving) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp, color = AppColors.Primary)
                } else {
                    Text("Add", color = if (isValid) AppColors.Primary else AppColors.TextSecondary)
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = !isSaving) {
                Text("Cancel", color = AppColors.TextSecondary)
            }
        }
    )
}

@Composable
private fun PatientCard(patient: Patient, onClick: () -> Unit) {
    val addedStr = remember(patient.createdAt) {
        SimpleDateFormat("MMM d, yyyy", Locale.getDefault()).format(Date(patient.createdAt))
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = AppColors.Surface),
        border = BorderStroke(1.dp, AppColors.Border),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(AppColors.PrimaryLight),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.Person,
                    contentDescription = null,
                    tint = AppColors.Primary,
                    modifier = Modifier.size(20.dp)
                )
            }
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = patient.name,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Medium,
                    color = AppColors.TextPrimary
                )
                Spacer(modifier = Modifier.height(2.dp))
                // Secondary identifying detail so two similarly-named
                // patients aren't indistinguishable in the list — picking
                // the wrong one here misattributes every future grade.
                Text(
                    text = "Patient ID ${patient.id} · Added $addedStr",
                    fontSize = 12.sp,
                    color = AppColors.TextSecondary
                )
            }
            Icon(
                imageVector = Icons.Filled.ChevronRight,
                contentDescription = null,
                tint = AppColors.TextSecondary,
                modifier = Modifier.size(18.dp)
            )
        }
    }
}