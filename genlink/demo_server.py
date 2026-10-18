"""Local offline demonstration UI; all mapping runs in the loaded engine."""
import argparse,json,time,threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from .retrieval_cli import MappingEngine

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--port',type=int,default=8765);p.add_argument('--weights-url',default='')
    a=p.parse_args();engine=MappingEngine(a.artifacts,device='cpu');lock=threading.Lock()
    validation=json.loads((a.artifacts/'models/VALIDATION_REPORT.json').read_text())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,payload,content_type='application/json'):
            data=payload.encode() if isinstance(payload,str) else json.dumps(payload).encode()
            self.send_response(200);self.send_header('Content-Type',content_type+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/validation':return self.send(validation)
            if self.path=='/links':return self.send(dict(weights=a.weights_url,github='https://github.com/vorunmosur/TeamStrawhats_hilabs'))
            return self.send(Path(__file__).with_name('demo.html').read_text(encoding='utf-8'),'text/html')
        def do_POST(self):
            try:
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                start=time.monotonic()
                with lock:result=engine.predict(data['entity'],data.get('context',''))
                self.send(dict(systems=result,seconds=time.monotonic()-start))
            except Exception as error:self.send(dict(error=str(error)))
    print(f'Offline TermMap demo ready at http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
