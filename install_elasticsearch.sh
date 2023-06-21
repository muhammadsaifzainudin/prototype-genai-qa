sudo apt install update

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.8.1-amd64.deb
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.8.1-amd64.deb.sha512
shasum -a 512 -c elasticsearch-8.8.1-amd64.deb.sha512 
sudo dpkg -i elasticsearch-8.8.1-amd64.deb

sudo systemctl enable elasticsearch.service
sudo systemctl start elasticsearch.service

##sudo nano /etc/elasticsearch/elasticsearch.yml