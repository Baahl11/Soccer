import React from 'react';
import {AbsoluteFill, interpolate, Sequence, spring, useCurrentFrame, useVideoConfig} from 'remotion';

const palette = {
  bg: '#050b11', panel: '#0a1722', line: '#17364a', text: '#eef7fb', muted: '#7d97aa', green: '#5de3b0', blue: '#78c8ff', amber: '#f2c66d'
};

const safe = (value, fallback='N/V') => value === null || value === undefined || value === '' ? fallback : String(value);
const localeCopy = (props) => {
  const locale = props.locale === 'es' ? 'es' : 'en';
  if (props.copy?.[locale]) return props.copy[locale];
  return props.copy || {};
};

const Brand = () => <div style={{fontSize:34,fontWeight:900,letterSpacing:-1.5}}>Soccer <span style={{color:palette.green}}>Edge</span></div>;

const Shell = ({children}) => (
  <AbsoluteFill style={{background:`radial-gradient(circle at 15% 0%, #123149 0%, ${palette.bg} 42%)`,color:palette.text,fontFamily:'Arial, Helvetica, sans-serif',padding:'120px 82px 150px'}}>
    {children}
  </AbsoluteFill>
);

const Chip = ({children, tone='blue'}) => <span style={{display:'inline-block',border:`2px solid ${tone==='green'?palette.green:tone==='amber'?palette.amber:palette.blue}`,borderRadius:999,padding:'10px 16px',fontSize:22,fontWeight:900,color:tone==='green'?palette.green:tone==='amber'?palette.amber:palette.blue,textTransform:'uppercase'}}>{children}</span>;

const Metric = ({label,value,accent=false}) => (
  <div style={{border:`2px solid ${palette.line}`,background:palette.panel,borderRadius:24,padding:'30px 28px'}}>
    <div style={{fontSize:22,color:palette.muted,textTransform:'uppercase',letterSpacing:2,fontWeight:800}}>{label}</div>
    <div style={{fontSize:70,fontWeight:950,marginTop:10,color:accent?palette.green:palette.text}}>{safe(value)}</div>
  </div>
);

const Intro = ({props}) => {
  const frame=useCurrentFrame(); const {fps}=useVideoConfig();
  const copy=localeCopy(props); const facts=props.facts||{};
  const enter=spring({frame,fps,config:{damping:18,stiffness:110}});
  return <div style={{opacity:enter,transform:`translateY(${interpolate(enter,[0,1],[45,0])}px)`}}>
    <Brand/>
    <div style={{marginTop:250,fontSize:30,color:palette.green,fontWeight:900,letterSpacing:4}}>MODEL VS MARKET</div>
    <div style={{fontSize:82,lineHeight:1.02,fontWeight:950,letterSpacing:-4,marginTop:24}}>{safe(copy.hook,'Soccer Edge · Model vs Market')}</div>
    <div style={{marginTop:52,fontSize:34,color:palette.muted,fontWeight:800}}>{safe(facts.home)} <span style={{color:palette.text}}>vs</span> {safe(facts.away)}</div>
  </div>;
};

const MarketPanel = ({props}) => {
  const frame=useCurrentFrame(); const {fps}=useVideoConfig(); const facts=props.facts||{};
  const enter=spring({frame,fps,config:{damping:18}});
  return <div style={{opacity:enter}}>
    <Brand/>
    <div style={{marginTop:185,display:'flex',justifyContent:'space-between',alignItems:'flex-start',gap:30}}>
      <div><div style={{fontSize:26,color:palette.muted,fontWeight:800}}>{safe(facts.league,'SOCCER EDGE')}</div><div style={{fontSize:54,fontWeight:950,marginTop:10}}>{safe(facts.home)} vs {safe(facts.away)}</div><div style={{fontSize:34,color:palette.blue,marginTop:20,fontWeight:850}}>{safe(facts.market)} {safe(facts.selection,'')}</div></div>
      <Chip tone="amber">{safe(facts.execution_status,'WATCH')}</Chip>
    </div>
    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:24,marginTop:90}}>
      <Metric label="Market fair" value={facts.market_probability_display}/>
      <Metric label="Soccer Edge" value={facts.model_probability_display} accent/>
    </div>
    <div style={{marginTop:24}}><Metric label="Probability gap" value={facts.edge_display} accent/></div>
    <div style={{marginTop:46,fontSize:25,color:palette.muted}}>Price: <b style={{color:palette.text}}>{safe(facts.price)}</b> · Book: <b style={{color:palette.text}}>{safe(facts.bookmaker)}</b></div>
  </div>;
};

const Decision = ({props}) => {
  const facts=props.facts||{}; const copy=localeCopy(props); const frame=useCurrentFrame(); const {fps}=useVideoConfig();
  const enter=spring({frame,fps,config:{damping:16}}); const blockers=Array.isArray(facts.blockers)?facts.blockers:[];
  return <div style={{opacity:enter}}>
    <Brand/>
    <div style={{marginTop:200,fontSize:28,color:palette.green,fontWeight:900,letterSpacing:3}}>EVIDENCE FIRST</div>
    <div style={{fontSize:74,fontWeight:950,letterSpacing:-3,marginTop:24}}>{safe(facts.execution_status,'N/V')}</div>
    <div style={{marginTop:55,border:`2px solid ${palette.line}`,background:palette.panel,borderRadius:28,padding:34}}>
      <div style={{fontSize:24,color:palette.muted,textTransform:'uppercase',fontWeight:900}}>Verified context</div>
      <div style={{fontSize:35,lineHeight:1.35,fontWeight:800,marginTop:20}}>{safe(facts.reason, blockers[0] || 'No additional verified context.')}</div>
      {blockers.slice(0,2).map((b,i)=><div key={i} style={{fontSize:27,color:palette.amber,marginTop:18}}>• {safe(b)}</div>)}
    </div>
    <div style={{fontSize:25,lineHeight:1.5,color:palette.muted,marginTop:45}}>{safe(copy.voiceover,'')}</div>
  </div>;
};

const Outro = ({props}) => {
  const frame=useCurrentFrame(); const opacity=interpolate(frame,[0,20],[0,1],{extrapolateRight:'clamp'});
  return <div style={{opacity,textAlign:'center',paddingTop:380}}><Brand/><div style={{fontSize:76,fontWeight:950,letterSpacing:-4,marginTop:100}}>Model vs Market.</div><div style={{fontSize:34,color:palette.muted,marginTop:32}}>Transparent evidence. No forced picks.</div><div style={{marginTop:80}}><Chip tone="green">soccer edge</Chip></div></div>;
};

export const SoccerEdgeShort = (props) => (
  <Shell>
    <Sequence from={0} durationInFrames={180}><Intro props={props}/></Sequence>
    <Sequence from={180} durationInFrames={330}><MarketPanel props={props}/></Sequence>
    <Sequence from={510} durationInFrames={240}><Decision props={props}/></Sequence>
    <Sequence from={750} durationInFrames={150}><Outro props={props}/></Sequence>
  </Shell>
);

export const SoccerEdgeShortCard = (props) => {
  const facts=props.facts||{};
  return <AbsoluteFill style={{background:`linear-gradient(135deg,#0a1e2b,${palette.bg})`,color:palette.text,fontFamily:'Arial, Helvetica, sans-serif',padding:'74px 90px'}}>
    <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}><Brand/><Chip tone="green">MODEL VS MARKET</Chip></div>
    <div style={{fontSize:52,fontWeight:950,marginTop:80}}>{safe(facts.home)} vs {safe(facts.away)}</div>
    <div style={{fontSize:28,color:palette.blue,fontWeight:850,marginTop:15}}>{safe(facts.market)} {safe(facts.selection,'')}</div>
    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:22,marginTop:62}}><Metric label="Market" value={facts.market_probability_display}/><Metric label="Model" value={facts.model_probability_display} accent/><Metric label="Gap" value={facts.edge_display} accent/></div>
    <div style={{marginTop:55,fontSize:24,color:palette.muted}}>Status: <b style={{color:palette.text}}>{safe(facts.execution_status)}</b> · Price: <b style={{color:palette.text}}>{safe(facts.price)}</b></div>
  </AbsoluteFill>;
};
