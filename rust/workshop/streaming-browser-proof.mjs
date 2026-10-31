// Exercise the shipped streaming client against a deliberately fragmented HTTP
// stream. Borg acceptance separately proves the exact native generation route.
import {createRequire} from 'node:module';
import {createServer} from 'node:http';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const require=createRequire('/home/muttleydosomething/hailstorm/frontend/hailstorm-frontend/package.json');
const {chromium,expect:baseExpect}=require('@playwright/test');
const expect=baseExpect.configure({timeout:5000});
const [modulePath,proof]=process.argv.slice(2);
await mkdir(proof,{recursive:true});
const wav=Buffer.alloc(44+48000*2);
wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);
wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);
wav.writeUInt32LE(24000,24);wav.writeUInt32LE(48000,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(wav.length-44,40);
for(let n=0;n<48000;n++)wav.writeInt16LE(Math.round(Math.sin(n*2*Math.PI*220/24000)*6000),44+n*2);
const audio=index=>({type:'audio',index,offset_samples:index*48000,samples:48000,wav:wav.toString('base64')});
const start={type:'start',version:1,format:'wav',sample_rate:24000};
const complete={type:'complete',frames:2,samples:96000,first_audio_ms:5};
const closed=[];
const server=createServer(async(req,res)=>{
 if(req.url==='/client.js'){res.setHeader('content-type','text/javascript');return res.end(await readFile(modulePath));}
 if(req.url==='/tts/stream'){
  let raw='';for await(const chunk of req)raw+=chunk;
  const {mode}=JSON.parse(raw);res.setHeader('content-type','application/x-ndjson');
  res.on('close',()=>closed.push(mode));
  const send=item=>res.write(JSON.stringify(item)+'\n');
  send(start);
  // Split even the base64 payload across network buffers; transport packet
  // boundaries must never be confused with the protocol's complete frames.
  const first=JSON.stringify(audio(0))+'\n';
  res.write(first.slice(0,17));res.write(first.slice(17,137));res.write(first.slice(137));
  setTimeout(()=>{
   if(res.destroyed)return;
   if(mode==='failure')send({type:'error',detail:'Native speech generation failed'});
   else if(mode==='order')send(audio(3));
   else if(mode!=='eof'){send(audio(1));send(complete);}
   res.end();
  },mode==='cancel'?800:180);return;
 }
 res.setHeader('content-type','text/html');res.end('<button id="generate-btn">Generate</button><textarea id="text"></textarea><script type="module">import {createSpeechStream} from "/client.js";window.player=createSpeechStream();window.frames=[];window.completed=[];document.addEventListener("myth-stream-frame",e=>window.frames.push(e.detail));document.addEventListener("myth-stream-complete",e=>window.completed.push(e.detail));window.run=mode=>{window.result=null;window.player.generate({mode}).then(async b=>window.result={ok:true,bytes:b.size},e=>window.result={ok:false,error:e.message,name:e.name});};document.querySelector("#generate-btn").onclick=()=>window.run(window.mode||"success");</script>');
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({headless:true});
const evidence={status:'running',started_unix_ms:Date.now(),cases:{}};
try{
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:'+server.address().port);
 await page.waitForFunction(()=>!!window.player);
 await page.locator('#generate-btn').click();
 await expect.poll(()=>page.evaluate(()=>window.frames.length)).toBe(1);
 assert.equal(await page.evaluate(()=>window.result),null,'first playback precedes transport completion');
 await page.waitForFunction(()=>window.result);
 const success=await page.evaluate(()=>({result:window.result,frames:window.frames,completed:window.completed}));
 assert(success.result.ok);assert.equal(success.result.bytes,44+96000*2);
 assert.equal(success.frames.length,2);assert.equal(success.frames[1].gap_seconds,0,'sample timeline remains contiguous when producer keeps up');
 evidence.cases.continuous=success;
 for(const mode of ['failure','order','eof']){
  await page.evaluate(mode=>{window.mode=mode;window.frames=[];window.completed=[];},mode);
  await page.locator('#generate-btn').click();await page.waitForFunction(()=>window.result);
  const result=await page.evaluate(()=>({result:window.result,frames:window.frames,completed:window.completed,stop_hidden:document.querySelector('#stream-stop-btn').hidden,status:document.querySelector('#stream-status').textContent}));
  assert(!result.result.ok,mode);assert.equal(result.completed.length,0);assert(result.stop_hidden);
  evidence.cases[mode]=result;
 }
 await page.evaluate(()=>{window.mode='cancel';window.frames=[];window.completed=[];});
 await page.locator('#generate-btn').click();await expect.poll(()=>page.evaluate(()=>window.frames.length)).toBe(1);
 await page.locator('#stream-stop-btn').click();await page.waitForFunction(()=>window.result);
 assert.equal((await page.evaluate(()=>window.result)).name,'AbortError');
 await page.waitForTimeout(900);
 assert.equal(await page.evaluate(()=>window.frames.length),1,'late old packets cannot schedule stale audio');
 assert.equal(await page.evaluate(()=>window.completed.length),0);
 evidence.cases.cancel={result:await page.evaluate(()=>window.result),closed:closed.includes('cancel')};
 assert(evidence.cases.cancel.closed,'cancel aborts HTTP stream');
 // The next logical turn owns fresh decoder/playback state.
 await page.evaluate(()=>{window.mode='success';window.frames=[];window.completed=[];});
 await page.locator('#generate-btn').click();await page.waitForFunction(()=>window.result);
 assert((await page.evaluate(()=>window.result)).ok);
 evidence.cases.reentry=await page.evaluate(()=>({result:window.result,frames:window.frames}));
 await page.screenshot({path:path.join(proof,'streaming-client.png')});
 assert.deepEqual(errors,[]);evidence.browser_errors=errors;evidence.status='passed';
}catch(error){evidence.status='failed';evidence.error=error.stack;throw error;}
finally{evidence.finished_unix_ms=Date.now();evidence.duration_ms=evidence.finished_unix_ms-evidence.started_unix_ms;await writeFile(path.join(proof,'browser-client.json'),JSON.stringify(evidence,null,2)+'\n');await browser.close();await new Promise(resolve=>server.close(resolve));console.log(JSON.stringify({status:evidence.status,proof}));}
