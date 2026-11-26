import concurrent.futures, pathlib, urllib.request, re, json, hashlib
ROOT=pathlib.Path(__file__).resolve().parent
URLS=['https://cloud.r-project.org/bin/windows/base/',
      'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/base/',
      'https://cran.rstudio.com/bin/windows/base/']
def get(url):
    try:
        text=urllib.request.urlopen(url,timeout=25).read().decode()
        return {'url':url,'exe':re.findall(r'href="([^"]+\.exe)"',text),'html':text}
    except Exception as e:return {'url':url,'error':str(e)}
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        result=list(pool.map(get,URLS))
    (ROOT/'download_sources.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in result: print({k:v for k,v in r.items() if k!='html'},flush=True)
