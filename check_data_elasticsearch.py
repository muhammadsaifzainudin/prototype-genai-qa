from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")

es.info().body

index_name = "book_wookieepedia_mpnet"
query_body = {
    'match': {
        'text': 'Ahsoka Tano'
    }
}
resp = es.search(index=index_name, query=query_body)

print("Got %d Hits:" % resp['hits']['total']['value'])
for hit in resp['hits']['hits']:
    print(hit["_source"]['text'])