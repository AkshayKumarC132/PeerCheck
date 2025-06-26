from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0009_feedback_created_at_feedback_created_by_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='SpeakerProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('embedding', models.JSONField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
    ]

