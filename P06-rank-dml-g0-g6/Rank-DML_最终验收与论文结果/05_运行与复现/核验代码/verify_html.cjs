const fs=require('fs');
const path=require('path');
const {pathToFileURL}=require('url');
const {chromium}=require('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/playwright');
const out=path.resolve(__dirname,'../..');
function walk(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(dir,e.name)):[path.join(dir,e.name)]);}
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1050},deviceScaleFactor:1});
 const results=[];const previews=path.join(out,'06_论文图表','预览');
 for(const file of walk(out).filter(x=>x.endsWith('.html'))){
   await page.goto(pathToFileURL(file).href,{waitUntil:'load'});
   const state=await page.evaluate(()=>({title:document.title,bodyWidth:document.body.scrollWidth,viewport:innerWidth,brokenImages:[...document.images].filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.src),headings:document.querySelectorAll('h1').length}));
   results.push({file,...state});
   if(file===path.join(out,'00_总览与验收','README.html'))await page.screenshot({path:path.join(previews,'HTML_readme.png')});
   if(file.endsWith('最终交付验收报告.html'))await page.screenshot({path:path.join(previews,'HTML_final_report.png')});
   if(file.endsWith('四个科学问题的实验结论.html')){
     await page.locator('img').first().scrollIntoViewIfNeeded();
     await page.screenshot({path:path.join(previews,'HTML_scientific_figure.png')});
   }
 }
 await browser.close();
 fs.writeFileSync(path.join(out,'05_运行与复现','HTML显示核验.json'),JSON.stringify(results,null,2),'utf8');
 const failures=results.filter(r=>r.brokenImages.length||r.headings!==1||r.bodyWidth>r.viewport+2);
 console.log(JSON.stringify({reports:results.length,failures}));
 if(failures.length)process.exitCode=1;
})();
