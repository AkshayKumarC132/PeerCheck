# Generated by Django 5.1.6 on 2025-05-29 09:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0006_alter_audiofile_transcription'),
    ]

    operations = [
        migrations.AddField(
            model_name='session',
            name='status',
            field=models.CharField(choices=[('active', 'Active'), ('archived', 'Archived')], default='active', max_length=20),
        ),
        migrations.AlterField(
            model_name='auditlog',
            name='action',
            field=models.CharField(choices=[('audio_upload', 'Audio Upload'), ('feedback_submit', 'Feedback Submit'), ('sop_create', 'SOP Create'), ('sop_update', 'SOP Update'), ('review_submit', 'Review Submit'), ('session_status_update', 'Session Status Update')], max_length=50),
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='role',
            field=models.CharField(choices=[('admin', 'Admin'), ('operator', 'Operator'), ('reviewer', 'Reviewer')], default='operator', max_length=20),
        ),
    ]
