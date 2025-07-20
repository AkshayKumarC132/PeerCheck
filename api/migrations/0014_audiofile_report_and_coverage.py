from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0013_alter_auditlog_action_referencedocument_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='audiofile',
            name='report_path',
            field=models.CharField(max_length=255, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='audiofile',
            name='coverage',
            field=models.FloatField(null=True, blank=True),
        ),
    ]
