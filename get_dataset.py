import paramiko
import subprocess
import csv
import os

# SSH connection details
ssh_host = 'is-garmin.haifa.ac.il'
ssh_user = 'barak'
ssh_password = 'Ry@5y7'

# MySQL command to retrieve all tables
mysql_command = 'mysql --user=root --password=R119p8#7 liat_db -e "SHOW TABLES;"'

# Establish SSH connection
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname=ssh_host, username=ssh_user, password=ssh_password)

# Execute MySQL command over SSH
stdin, stdout, stderr = ssh_client.exec_command(mysql_command)

# Read the output of MySQL command
output = stdout.read().decode()

# Split the output into table names
table_names = output.strip().split('\n')[1:]

# Loop through each table and export to CSV
for table_name in table_names:
    csv_file = f'{table_name}.csv'
    mysql_export_command = f'mysql --user=root --password=R119p8#7 liat_db -e "SELECT * FROM {table_name};" | sed \'s/\\t/","/g;s/^/"/;s/$/"/\' > {csv_file}'
    ssh_client.exec_command(mysql_export_command)
    sftp = ssh_client.open_sftp()
    sftp.get(csv_file, os.path.join(os.getcwd(), csv_file))
    sftp.close()

# Close the SSH connection
ssh_client.close()