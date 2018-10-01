cp -r /opt/ga_epna /opt/code
cd /opt/code
git pull
python /opt/code/ingestion/connector/ga_epna_connector.py
