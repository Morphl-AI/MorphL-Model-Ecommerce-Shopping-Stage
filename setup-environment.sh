echo 'Installing the Shopping Stage Prediction module...'
echo

# Create file with env variables for the module
touch /home/airflow/.morphl_ga_epna_environment.sh
chmod 660 /home/airflow/.morphl_ga_epna_environment.sh
chown airflow /home/airflow/.morphl_ga_epna_environment.sh
truncate -s 0 /home/airflow/.morphl_ga_epna_environment.sh

echo "export GA_EPNA_KEY_FILE_LOCATION=/opt/secrets/ga_epna/service_account.json" >> /home/airflow/.morphl_ga_epna_environment.sh
echo "export GA_EPNA_VIEW_ID=\$(</opt/secrets/ga_epna/viewid.txt)" >> /home/airflow/.morphl_ga_epna_environment.sh

# Add credentials to airflow profile
if ! grep -q "/home/airflow/.morphl_ga_epna_environment.sh" /home/airflow/.profile; then
  echo ". /home/airflow/.morphl_ga_epna_environment.sh" >> /home/airflow/.profile
fi

# Set file permissions
chmod -R 775 /opt/secrets/ga_epna
chmod 660 /opt/secrets/ga_epna/viewid.txt
chmod 660 /opt/secrets/ga_epna/service_account.json
chgrp airflow /opt/secrets/ga_epna /opt/secrets/ga_epna/service_account.json /opt/secrets/ga_epna/viewid.txt