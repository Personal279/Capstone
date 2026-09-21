package com.pes.facialparalysis.reminders

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Fires a locally-scheduled reminder notification and, if recurring, re-arms the next occurrence. */
class ReminderAlarmReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val reminderId = intent.getIntExtra("reminder_id", -1)
        val title = intent.getStringExtra("title") ?: "Reminder"
        val notes = intent.getStringExtra("notes") ?: ""
        val hour = intent.getIntExtra("hour", 9)
        val minute = intent.getIntExtra("minute", 0)
        val daysCsv = intent.getStringExtra("days_csv") ?: ""

        NotificationHelper.show(context, notificationId = reminderId, title = title, body = notes.ifBlank { "Scheduled reminder" })

        if (daysCsv.isNotBlank()) {
            ReminderScheduler.rescheduleRecurring(context, reminderId, title, notes, hour, minute, daysCsv)
        }
    }
}
