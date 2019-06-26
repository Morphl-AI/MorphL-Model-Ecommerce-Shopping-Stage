echo 'Installing the Shopping Stage Prediction module...'
echo

touch /home/airflow/.morphl_ga_epna_environment.sh
chmod 660 /home/airflow/.morphl_ga_epna_environment.sh
chown airflow /home/airflow/.morphl_ga_epna_environment.sh

echo "export GA_EPNA_KEY_FILE_LOCATION=/opt/secrets/ga_epna/gcloud_service_account.json" >> /home/airflow/.morphl_ga_epna_environment.sh
echo "export GA_EPNA_VIEW_ID=\$(</opt/secrets/ga_epna/viewid.txt)" >> /home/airflow/.morphl_ga_epna_environment.sh
echo ". /home/airflow/.morphl_ga_epna_environment.sh" >> /home/airflow/.profile
