#%%
import re, time, json, os
from io import StringIO

from dataclasses import dataclass

import openai
from dotenv import load_dotenv

import pandas as pd
#%%
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
COL_MEDPROB = "Medical problem that event is related to"
COL_EVENT = "Clinical event"
COL_DATE = "Event date (or 'N/A' if not mentioned)"
COL_REASON = "Reasoning for 'Event date' output"
#%%
@dataclass
class GPTResponse:
    date: str
    string: str
    obj: dict

def queryGPT(query, system_query = None, model = "gpt-3.5-turbo"):
    # model = "gpt-4"

    output_raw = None
    messages = [{"role": "user", "content": query},]
    if system_query:
        messages.insert(0, {"role": "system", "content": F"{system_query}"})

    print(messages)
    try:
        output_raw = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            n=1
            )
        if output_raw != None:
            print(F"Received response")

    except (openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.RateLimitError) as e:
        print(f"OpenAI API error: {e}\nRetrying...")
        time.sleep(1.5)
        output_raw = queryGPT(query=query, system_query=system_query, model=model)
        
    return output_raw


doc_list: list[str] = []
#%% Appending docs
doc_list.append("""Document date: June 19, 2023
Document type: Progress note

Subjective:
Garnett Groves is a 71yo male from LTC who presented on June 17th with "stroke symptoms" (aLOC, ?dysarthria, RT sided weakness) and hyperglycemia (BG 55)

1. Hyperglycemia secondary to T2D
- History of T2D, not on anti-hyperglycemics when admitted to LTCs
- A per patient, was taken off of diabetes medication 3 years ago as his doctor told him that "he doesn't need it anymore"
- A1c in Nov 2022 was 5.4%, most recent A1c (June 18th) at 10.9%
- Started insulin therapy on admission
- Last B-hydroxybutyrate 0.3, No evidence of DKA
- POC glucose levels still elevated at ~20


2. aLOC & dysarthria
- Was noted to have slurred speech and RT sided weakness on June 16
- According to brother and patient himself, his speech is dysarthric at baseline
- CT head on admission showed encephalomalacia vs edema in LT parietal area
- History of epilepsy, carbamazepine levels normal


3. Hemochromatosis with cirrhosis findings
- HFE genetic test results were positive as per GP
- Ferritin levels June 18th at 1320 (H)
- Previous therapeutic phlebotomy

4. Bicytopenia
- Anemia: Hb decreased from 119 (June 16) to 108 (June 18) to 99 (June 19), MCV 103
- Thrombocytopenia: Plt decreased from 102 (June 16) to 80 (June 18) to 76 (June 19)
- Patient does not endorse any dizziness or lightheadedness

Assessment and Plan:
In summary, this is a 71yo male from LTC who presented on June 17th with "stroke symptoms" and hyperglycemia

1. Hyperglycemia secondary to T2D
- T2D was previously well managed so anti-hyperglycemics were discontinued, patient possibly lost to follow-up resulting in interval worsening of T2D management
- BGs still suboptimal on current inpatient diabetes regime
- Insulin Aspart baseline increased to 7IU from 5IU
- Insulin Glargine HS increased to 25IU from 20IU
- Will continue to gradually increase insulin and optimize insulin regime

2. aLOC & dysarthria
- MRI grossly normal with early neurodegenerative changes and old infarction in LT posterior parietal cortex consistent with earlier CT
- Dysarthria is chronic but SLP investigation ongoing
- During exam today, pt was AOx3 with intact cognition, overall appears to be have intermittent periods of confusion which could possibly be a deterioration from his baseline

3. Hemochromatosis with cirrhosis findings
- Previous therapeutic phlebotomy but currently anemic
- Outpatient Echo to assess for cardiac Fe deposition
- Needs consult to non-malignant heme for further management (potentially Dr. Yee who knows of the patient)

4. Bicytopenia
- RBCs and Plts have been trending downwards since admission
- Anemia may be secondary to iron loading
- B12 normal, last set of liver enzymes were WNL
- Possibly due to chronic cirrhosis but unable to find historical CBCs on clinicalconnect to corroborate

5. Housekeeping
- Falls: Patient is falls risk
- PT and OT consulted for functional status and to assess for transfers
- DVT prophylaxis: DVTp ordered
- Code status: Per patient's sister-in-law, patient is full code
- Barriers to discharge: BGs, OT/PT clearance
- Disposition: Plan to discharge back to LTC
- Followup: Pending further investigations and optimized glycemic control
""")

doc_list.append("""Document date: June 20, 2023
Document type: Progress note

Subjective:
Gardnett Groves is a 71yo male from LTC who presented on June 17th with "stroke-like symptoms" and hyperglycemia.

1. Hyperglycemia secondary to T2D
- History of T2D, not on anti-hyperglycemics when admitted to LTCs
- As per patient, was taken off of diabetes medication 3 years ago as his doctor told him that "he doesn't need it anymore"
- A1c in Nov 2022 was 5.4%, most recent A1c (June 18th) at 10.9%, admission BG was 55
- Last B-hydroxybutyrate 0.3, No evidence of DKA
- Started insulin therapy on admission, last doses of insulin were 25IU long active, 7IU TID with meals short acting baseline
- POC glucose levels in AM today at 18.5 still elevated at ~20


2. aLOC & dysarthria
- Was noted to have slurred speech and RT sided weakness on June 16
- According to brother and patient himself, his speech is dysarthric at baseline
- CT head on June 16 and MRI on June 20 showed old ischemic changes in LT posterior parietal cortex
- History of epilepsy, carbamazepine levels normal
- Overall feeling better today 


3. Hemochromatosis with cirrhosis findings
- HFE genetic test results were positive as per GP
- Ferritin levels June 18th at 1320 (H)
- Previous therapeutic phlebotomy
- Last set of liver enzymes were normal but albumin was trending down
- Nursing team noted worsening pitting edema, increased to 2+ from 1+

4. Bicytopenia
- Anemia: Hb decreased from 119 (June 16) to 108 (June 18) to 99 (June 19), MCV 103
- Thrombocytopenia: Plt decreased from 102 (June 16) to 80 (June 18) to 76 (June 19)
- Patient does not endorse any dizziness or lightheadedness

Assessment and Plan:
In summary, this is a 71yo male from LTC who presented on June 17th with "stroke symptoms" and hyperglycemia

1. Hyperglycemia secondary to T2D
- T2D was previously well managed so anti-hyperglycemics were discontinued, patient possibly lost to follow-up resulting in interval worsening of T2D management
- BGs suboptimal on current inpatient diabetes regime
- Insulin Aspart baseline increased to 7IU TID from 5IU June 19
- Insulin Glargine HS increased to 35IU from 25IU June 20
- Will continue to gradually increase insulin and optimize insulin regime
- Once sugars are controlled, he can be discharged back to LTC, hopefully by the end of the week 

2. aLOC & dysarthria
- No acute changes noted on CT and MRI head
- Dysarthria is chronic, SLP recommends diet to be minced, no bread/thin liquids NO STRAWS and meds to be crushed in applesauce 
- Overall appears to be have intermittent periods of confusion which could possibly be a deterioration from his baseline

3. Hemochromatosis with cirrhosis findings
- HFE genetic test results were positive as per GP
- Previous therapeutic phlebotomy but currently anemic
- Outpatient Echo to assess for cardiac Fe deposition
- Consulted to non-malignant heme for further management (potentially Dr. Yee who knows of the patient)
- Repeat albumin for worsening of edema 

4. Bicytopenia
- RBCs and Plts have been trending downwards since admission
- B12 normal, last set of liver enzymes were WNL
- Possibly due to chronic cirrhosis but unable to find historical CBCs on clinicalconnect to corroborate
- Anemia may be secondary to iron loading
- Repeat CBCs to trend 

5. Housekeeping
- D/C'd foley 
- Falls: Patient is falls risk
- PT and OT consulted for functional status and to assess for transfers
- DVT prophylaxis: DVTp ordered
- Code status: Per patient's sister-in-law, patient is full code
- Barriers to discharge: BGs, OT/PT clearance
- Disposition: Plan to discharge back to LTC
- Followup: Pending further investigations and optimized glycemic control 
""")
doc_list.append("""Document date: June 21, 2023
Document type: Progress note

Subjective:
Gardnett Groves is a 71yo male from LTC who presented on June 17th with "stroke-like symptoms" and hyperglycemia.

1. Hyperglycemia secondary to T2D
- History of T2D, not on anti-hyperglycemics when admitted to LTC
- As per patient, was taken off of diabetes medication 3 years ago as his doctor told him that "he doesn't need it anymore"
- A1c in Nov 2022 was 5.4%, most recent A1c (June 18th) at 10.9%, admission BG was 55
- Last B-hydroxybutyrate 0.3, No evidence of DKA
- Started insulin therapy on admission, last doses of insulin were 35IU long active, baseline 7IU TID with meals short acting (receiving ~37IU total throughout day)
- POC glucose levels in AM today at 14.1, most recent daytime BG at 19.4

2. UTI
- Nursing team noted foul smelling urine with sedimentation on June 20th
- Patient c/o pain on urination, no frequency/urgency, no flank pain 

2. aLOC & dysarthria
- Was noted to have slurred speech and RT sided weakness on June 16
- According to brother and patient himself, his speech is dysarthric at baseline
- CT head on June 16 and MRI on June 20 showed old ischemic changes in LT posterior parietal cortex
- History of epilepsy, carbamazepine levels normal
- Overall feeling better today 


3. Hemochromatosis with cirrhosis findings
- HFE genetic test results were positive as per GP
- Ferritin levels June 18th at 1320 (H)
- Previous therapeutic phlebotomy
- Last set of liver enzymes were normal 
- Albumin was trending down: 28 (June 21) <- 34 (June 17) <- 38 (June 16)
- Nursing team noted worsening pitting edema June 20th, increased to 2+ from 1+
- Patient also notes SOB when fully reclined but this has been the case for a month


4. Bicytopenia
- Anemia, Decreasing Hb: 101 (June 21) <- 99 (June 19) <- 108 (June 18) <- 119 (June 16)
- Thrombocytopenia: 84 (June 21) <- 76 (June 19) <- 80 (June 18) <- 102 (June 16) 
- Patient does not endorse any dizziness or lightheadedness

Assessment and Plan:
In summary, this is a 71yo male from LTC who presented on June 17th with "stroke symptoms" and hyperglycemia

1. Hyperglycemia secondary to T2D
- T2D was previously well managed so anti-hyperglycemics were discontinued, patient possibly lost to follow-up resulting in interval worsening of T2D management
- BGs suboptimal on current inpatient diabetes regime
- Insulin Aspart baseline increased to 7IU TID from 5IU June 19
- Insulin Glargine HS increased to 40IU from 35IU June 21
- Will continue to gradually increase insulin and optimize insulin regime
- Once sugars are controlled, he can be discharged back to LTC, hopefully by the end of the week 

2. UTI 
- Urinalysis turbid with 1.0 protein, positive nitrites, 4+ bacteria 
- Foley removed June 20th, foley culture pending 
- Likely uncomplicated lower UTI
- Started on nitrofuratoin 100mg PO BID x 5 days

2. aLOC & dysarthria
- No acute changes noted on CT and MRI head
- Dysarthria is chronic, SLP recommends diet to be minced, no bread/thin liquids NO STRAWS and meds to be crushed in applesauce 
- Overall appears to be have intermittent periods of confusion which could possibly be a deterioration from his baseline

3. Hemochromatosis with cirrhosis findings
- HFE genetic test results were positive as per GP
- Previous therapeutic phlebotomy but currently anemic
- Outpatient Echo to assess for cardiac Fe deposition
- Consulted to non-malignant heme for further management (potentially Dr. Yee who knows of the patient)
- Decreasing albumin likely nutritional rather than hepatic 

4. Bicytopenia
- RBCs and Plts have been trending downwards over first few days of admission but now seems to be stable 
- B12 normal, last set of liver enzymes were WNL
- Possibly due to chronic cirrhosis but unable to find historical CBCs on clinicalconnect to corroborate
- Anemia may be secondary to iron loading
- Continue to trend 

5. Housekeeping
- Falls: Patient is falls risk
- PT and OT consulted for functional status and to assess for transfers
- DVT prophylaxis: DVTp ordered
- Code status: Per patient's sister-in-law, patient is full code
- Barriers to discharge: BGs, OT/PT clearance
- Disposition: Plan to discharge back to LTC
- Followup: Pending further investigations and optimized glycemic control             
""")
doc_list.append("""Document date: June 22, 2023
Document type: Progress note

Subjective:
Garnett Groves is a 71yo male from LTC who presented on June 17th with "stroke-like symptoms" and hyperglycemia.

1. Hyperglycemia secondary to T2D
- History of T2D, not on anti-hyperglycemics when admitted to LTC
- As per patient, was taken off of diabetes medication 3 years ago as his doctor told him that "he doesn't need it anymore"
- A1c in Nov 2022 was 5.4%, most recent A1c (June 18th) at 10.9%, admission BG was 55
- Last B-hydroxybutyrate 0.3, No evidence of DKA
- Started insulin therapy on admission, last doses of insulin were 40IU long active, baseline 7IU TID with meals short acting (receiving ~35IU total throughout day)
- POC glucose levels in AM today at 11.7, most recent daytime BG at 15.9 which is slightly improved from yesterday. Will keep current regime

2. UTI
- Nursing team noted foul smelling urine with sedimentation on June 20th
- Patient c/o dysuria, no frequency/urgency, no flank pain
- Has received x2 doses of nitrofuratoin so far, but C/S returned Proteus mirabilis which was resistant to nitrofuratoin 


3. Bicytopenia
- Anemia, Hb has been stable: 106 (June 22) <- 101 (June 21) <- 99 (June 19) <- 108 (June 18)
- Thrombocytopenia, Plt has been stable: 93 (June 22) <- 84 (June 21) <- 76 (June 19) <- 80 (June 18)
- Patient does not endorse any dizziness or lightheadedness

Assessment and Plan:
In summary, this is a 71yo male from LTC who presented on June 17th with "stroke symptoms" and hyperglycemia

1. Hyperglycemia secondary to T2D
- T2D was previously well managed so anti-hyperglycemics were discontinued, patient possibly lost to follow-up resulting in interval worsening of T2D management
- BGs suboptimal on current inpatient diabetes regime
- Insulin Aspart baseline increased to 7IU TID from 5IU June 19
- Insulin Glargine HS increased to 40IU from 35IU June 21
- Will continue to gradually increase insulin and optimize insulin regime
- Once sugars are controlled (BGs within 10-12 range), he can be discharged back to LTC, hopefully by the end of the week 

2. UTI 
- Urinalysis turbid with 1.0 protein, positive nitrites, 4+ bacteria 
- Foley removed June 20th, foley culture pending 
- Likely uncomplicated lower UTI
- Started on nitrofuratoin 100mg PO BID June 21 but swapped to Amox-Clav 500-125mg BID June 22 as per pharmacist suggestion due to culture & sensitivity results


3. Bicytopenia
- RBCs and Plts have been trending downwards over first few days of admission
- B12 normal, last set of liver enzymes were WNL
- Possibly due to chronic cirrhosis but unable to find historical CBCs on clinicalconnect to corroborate
- Anemia may be secondary to iron loading
- Appears stable now

4. Housekeeping
- Falls: Patient is falls risk
- PT and OT consulted for functional status and to assess for transfers
- DVT prophylaxis: DVTp ordered
- Code status: Per patient's sister-in-law, patient is full code
- Barriers to discharge: 
	- Glucose control
- Disposition: Plan to discharge back to LTC
- Followup: Pending further investigations and optimized glycemic control 

Resolved issues:

2. aLOC & dysarthria
- No acute changes noted on CT and MRI head
- Dysarthria is chronic, SLP recommends diet to be minced, no bread/thin liquids NO STRAWS and meds to be crushed in applesauce 
- Overall appears to be having intermittent periods of confusion which could possibly be a deterioration from his baseline

3. Hemochromatosis with cirrhosis findings
- HFE genetic test results were positive as per GP
- Previous therapeutic phlebotomy but currently anemic
- Outpatient Echo to assess for cardiac Fe deposition
- Consulted to non-malignant heme for further management (potentially Dr. Yee who knows of the patient)
- Decreasing albumin likely nutritional rather than hepatic 
""")

doc_list.append("""
""")


doc_list.append("""Document date: April 25, 2023
Term infant born through SVD, with prolonged ROM, presenting with increased work of breathing at 7 minutes of life, requiring CPAP. Presentation is most consistent with TTN. No risk factors for RDS (no prematurity, not IDM). No meconium, so meconium aspiration unlikely. Normal oxygen saturation on room air, so unlikely congenital heart disease.

 Risk factors for sepsis include prolonged ROM (24hrs). No other risk factors present (no maternal fever, GBS negative). Therefore screening bloodwork has not been ordered, but 24hrs observation with vital signs is appropriate and will already be conducted in context of admission for TTN.

 Admit to NICU under Dr. Komsa
 Continue CPAP 6 room air and wean as tolerated
 CXR
 NPO
 OG tube
 Blood gas and glucose at 2 hrs of life
 If blood gas normal, clinically improved, can consider trial off CPAP later tonight and initiation of oral feeds
 If there is ongoing need for CPAP, or blood glucose at 2hrs of life is abnormal, we will start IVF.
 24 hr bili and NBS
""")

doc_list.append("""Document date: April 26, 2023
Subjective:
GA: 38+2days, Mother G4T2P0L1, 40yo at birth.
 Sebastian is a is a 20hr old term infant born through SVD with prolonged ROM but APGAR 9 at 1 and 5 min. Presenting with increased work of breathing at 7 minutes of life, requiring CPAP and admission to NICU. Presentation is most consistent with TTN.

 BW: 3.335kg, CW: None recorded

 1. Respiratory:
 - Some abdominal breathing and tachnypnea last night last, also had a brief episode of desaturation but otherwise no concerns overnight
 - Was just wean off of 5 cmH2O CPAP

 2. Nutrition:
 - Has been NPO since admission, NG tube inserted
 - Mother wants to start breastfeeding, currently on D10W at 8.3

 3. Infectious:
 - CXR clear
 - No fever

 4. Metabolic:
 - No VBG abnormalities
 - POC BG April 25 = 3.4
 - Bilirubin and NBS pending for today
 
Assessment/Plan 
Sebastian is a 20hr old term infant presenting with increased work of breathing at 7 minutes of life consistent with TTN, requiring CPAP and admission to NICU.

1. Respiratory:
 - Was on CPAP 5cmH2O, tachypnea resolved with RR ~50
 - Will trial off of CPAP this AM and monitor

2. Nutrition:
 - Stable and doing well so far, currently trialling off CPAP to initiate oral feeds
 - Mother wants to breastfeed, will wean off of D10W after initiating feeds 
 - Will taper D10W using NICU hypoglycemia IV weaning protocol; half D10W rate after first normal pre-prandial BG, then discontinue D10W after second normal pre-prandial BG

3. Infectious:
 - No worrying signs for infectious etiology

 4. Metabolic:
 - Bili and NBS ordered, to return today in PM


 Overall, his TTN has been resolved and he would be clear for discharge after he is successful weaned off of D10W and Bili + NBS return with no abnormalities.
 Will have a follow-up with GP in 1-3 days

""")

doc_list.append("""Document date: April 27, 2023
Subjective 
DOL: 2
 GA: 38+2
 CA: 38+4
 BW: 3.335 kg
 CW: 3.150 kg (-0.2 kg or -5.5%)

Sebastian is a is a 41hr old term infant born through SVD with prolonged ROM but APGAR 9 at 1 and 5 min. Presented to the NICU yesterday requiring CPAP.  Presentation at that time was most consistent with TTN.

Active Issues:
 1. Respiratory:
 - weaned off CPAP yesterday; currently on RA
 - was seen and assessed by Pediatrician on call yesterday to assess for three desaturations today. Yesterday he did not have apnea or bradys - rather he was breathing shallowly for each. screening CBC/CRP/gas to assess for signs of potential infection and all were reassuring at this time
 - As per nursing this morning there continued to be cluster episodes of bradycardia with desaturations and some associated periods of apnea, all of which self resolved.
 - RN mentioned some chewing between feeds although the cluster of apnea/desat/brady happened ~1 hour post-feed.

 2. Nutrition
 - is currently being entirely breastfed ad lib
 - NG tube was pulled yesterday; no longer has an IV since D10W was stopped yesterday.

 3. Infectious:
 - CXR yesterday was clear.
 - No fever at present
 - Risk factors for infection include prolonged rupture of membranes
 - no maternal fevers at the time of birth.

 4. Metabolic:
 - No concerns at present.
 - VBG, CRP, Bilirubin and NBS were all reassuring.

Assessment/Plan
Overall, Sebastian is a 41h old infant, who continues to experience periods of bradycardia with desaturation and an isolated period of apnea. Given the persistence of this presentation it may not appear to be in keeping with TTN. Potentially causes include - infectious, respiratory, cardiac, reflux, and neurologic. Thus we have elected to complete a further workup. Our plan for today is as follows.

 1. Respiratory - A/B's full workup
 - repeat CXR
 - repeat bloodwork - CBC, CRP, blood gases
 - work-up for consideration of potential cardiac cause - ECG and 4 limb BP
 - work-up for potentially infectious cause - blood cultures and empiric treatment with ampicillin, (66.75 mg= 0.33 vial, 50 mg/kg, IV-PIGGYBACK, q8h) and tobramycin, (13.34 mg= 0.33 mL, 4 mg/kg, IV-PIGGYBACK, q24h-ATC)

 2. Nutrition
 - continue with ad lib breastfeeding
 - start Florababy, 0.5 g= 1 packet, PO/FEEDING TUBE, Daily to help with microbiome health given the newly prescribed antibiotics
 - restart Dextrose 10% in Water intravenous solution 1,000 mL [60 mL/kg/day], 1000 mL, IV-CONTINUOUS

 3. Infectious
 - see respiratory workup as noted above
 - continue to monitor clinically

 4. Metabolic
 - no concerns at present

 Both parents (Mom: Christian and Dad: Alex) were present at rounds. They were counselled at length on our current management plan. All questions were answered at this time.

""")

doc_list.append("""Document date: April 28, 2023
Subjective 
GA: 38+2days, Mother G4T2P0L1, 40yo at birth.
 CA: 38+4days
 BW: 3.335kg, CW: 3.070kg (-7.8%), Weight on April 26th 3.150kg

Sebastian is a is a 2-day old term infant born through SVD with prolonged ROM but APGAR 9 at 1 and 5 min. Presenting with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN.

 1. Respiratory:
 - Weaned off CPAP April 26th; currently on RA
 - He has continued to have episodes of desaturations over the last few days. Yesterday he had a cluster over 15 minutes of apnea/brady/desats
 - Last night, he had x3 episodes of apnea/bradycardia/desaturations which ranged from requiring vigorous stimulation to resolving spontaneously
 - No reflux after feeding as per nurse
 - Mother feels that these episodes have been slightly improving since yesterday when he had cluster episodes of apnea/bradycardia/desaturation

 2. Infectious:
 - No fevers since birth (max 37.6C), no fevers today or overnight
 - Risk factors for infection include prolonged rupture of membranes

 3. Fluid/Nutrition
 - Restarted D10).2NaCl IV 1,000 mL [60 mL/kg/day], currently on D10+0.2NaCl in addition to breast feeding to hunger cues
 - Started probiotics on April 27 for dysbiosis prophylaxis in response to initiation of abx - parents chose to bring in BioGaia after informed discussion about the Florababy recall from a previous lot
 - Has been losing some weight (-7.8% loss from BW), currently breastfeeding q3h

Assessment/Plan 
Sebastian is a 2-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- apnea/bradycardia. Currently the differential remains open as we continue to investigate potential etiologies:

1. Respiratory:
 - He has continued to have episodes of desaturations +/- bradycardia/apnea over the last few days, although this has been improving as per mother
 - ECG and 4 limb BP April 27th normal, bradycardia also occurs after apnea/desaturation. Episodes unlikely to be cardiac in etiology
 - No reflux after feeding as per nurse and episodes are not clearly post-prandial. Episodes less likely to be due to reflux/aspiration
 - Repeat bloodwork (CBC, CRP, VBGs) normal, unlikely to have a metabolic cause
 - US head scheduled for IVH and neuro cause, results pending


 2. Infectious:
 - No fevers thus far, most recent repeat CXR was clear
 - PROM but otherwise no risk factors for sepsis
 - IV abx started on April 27 for 24hrs, on probiotics for dysbiosis prophylaxis
 - Clinically, patient is well, likely no underlying infectious etiology

 3. Fluid/Nutrition
 - On D100.2NaCl in addition to breast feeding q3h to hunger cues
 - Is also on adequate vitamin D supplementation (currently on 10mcg/day = 400iu)
 - Current weight is a -7.8% loss from BW but still within normal limits, continue current fluid/nutrition plan
 - Discussed the importance of feeding q3 hours - not to let him sleep for longer until he has regained his birth weight.

 Overall, we will await US head results, reassess after his 24hrs of abx, and continue to monitor his condition. If these episodes become worsen, we will consider consultation with MUMC for further investigation

""")

doc_list.append("""Document date: April 29, 2023
Subjective 
Term infant
 Prolonged ROM, Mom covered with Abx in labour
 TTN, resolving
 Shallow Resps and A&Bs NYD: Head US normal, WBC x2 normal , CRP peak 1.9, Babe on Abx, Cultures negative at 48hours, unremarkable cap gas

 -Last A&B 0400
 -Feeding not yet established, mom has not had much milk but was reluctant to supplement
 -Willing to start today, had LC assessment
 -Poor urine output over night
 -Normal lytes and creatinine
 -Bili 273, under phototherapy range, up in HRZ (Mom and babe are both blood Group O+ve)

Assessment/Plan 
1. Prolonged rupture of membranes 
Labs are reassuring but we have no explanation for A&Bs
 I will continue Abx for now and review depending on babe's clinical status in am


2. Respiratory distress 
No distress at basleine.
 Ongoing A&bs NYD, ? resolving TTN, consider metabolic or cardiac causes in differential
 Consult with Mac NICU if ongoing
  

Orders: 
Bilirubin-Neonatal
Communication Order
Communication Order
Rpt bili in am
Mom updated at bedside in am
""")


doc_list.append("""Document date: April 30, 2023
Subjective 
Day 5
 Term infant
 Prolonged ROM, Mom covered with Abx in labour
 TTN, resolving
 Shallow Resps and A&Bs NYD: Head US normal, WBC x2 normal , CRP peak 1.9, Babe on Abx, Cultures negative at 48hours, unremarkable cap gas
[1]

 -No A&Bs for over24hrs
 -Feeds have improvement as has urine output(1.8ml/kg/hr)
 -Breast feeding well, still needing bottle top ups
 -Wt 3070g up 25g
 -Bili 254, down ,below phototherapy range

Assessment/Plan 
1. Prolonged rupture of membranes 
Babe is doing better since Abx were started although wbcs and CRPs were normal
 Continue Antibiotics for a total of 5 days to cover the possibility of infection


2. Respiratory distress 
Distressed resolved
 A&Bs resolved


Orders: 

Communication Order

Tobramycin Level Trough

Continue working on feeds
 Tobra trough leve in am
 Mom updated at the bedside

""")
doc_list.append("""Document date: May 1, 2023
Subjective 
GA: 38w+2days, Mother G4T2P0L1, 40yo at birth.
 CGA: 39w
 BW: 3.335kg, CW: 3.135 (-6.0%)

 Sebastian is a is a 5-day old term infant born through SVD with APGAR 9 at 1 and 5 min who presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but she has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - Desaturation +/- bradycardia/apnea episodes were improving over the weekend (24hr stretch b/n April 29-30 with no episodes).
 - Unforunately had x4 desaturation episodes last night and this morning  without apnea and only bradycardia with the last episode.
 - Desaturations required NC O2 for a period, currently back on room air
 - According to nursing, these x4 episodes mostly occured when sleeping and once related to reflux
 - During these episodes, his breathing was noted to be very shallow and slow
 - Tone also noted to be relatively weak when sleeping

 2. Fluid and nutrition
 - No struggles with feeding, currently breastfeeding adlib with formula top-up and on 2mL/hr D10 0.2%NS
 - Weight has been trending upwards (3.135 April 30 <- 3.070 April 29)
 - Has continued to have mild reflux post-feeding
 - Tobramycin was held for April 29 due to low urinary output one day as she was lacking urination, has had x3 doses so far, one more scheduled today
 - Started probiotics on April 27 for dysbiosis prophylaxis in response to initiation of abx

 3. Hyperbilirubinemia
 - Initially elevated at 273 on April 29, then decreased to 254 on April 30.
 - Was subthreshold for both measurements so phototherapy was not indicated

Assessment/Plan 
Sebastian is a 2-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Currently the differential remains open as we continue to investigate potential etiologies:

 1. Respiratory
 - Presistent episodes of desaturation which has been improving slightly but has not resolved. No apneas for >48 hours
 - Continue abx as scheduled (ampicillin, tobramycin)
 - Extensive workup has been done (e.g., ECG, 4 limb BP, CXR, US head) to rule out cardiac, respiratory, neuro structural, GI, and metabolic etiologies of these episodes.
 - As all investigations done so far returned normal, we will consult MUMC for a opinion for additional workup

 2. Fluid/Nutrition
 - Feeding well but still has mild reflux that may be contributing to some episodes of desaturation
 - Antacids were brought up during our conversation with the parents, to be reassessed after discussion with Mcmaster
 - Plan to continue breastfeeding ad lib with formula top-up and D10-0.2%NS

 3. Hyperbilirubinemia
 - Bilirubin levels improving, no major risk factors
 - Will not need phototherapy
 
Addendum
  Spoke with McMaster NICU at 1300 (Dr Williams) and case reviewed. Based on Sebastian's course to date she agrees that an infectious process does not fully explain his ongoing desats and brady events. She advised that it would be reasonable to discontinue Abx tomorrow after a 5 day course is completed. She did note that if there are new apnea events she would recommend a repeat FSWU with an LP and antibiotic restart. She also notes that it seems unlikely that there is a cardiac component to these ongoing events as the infant is well between episodes, with no murmurs, normal 4 limb BPs and normal perfusion. She also was reassured that a HUS was normal and the infant has been cueing for feeds and is feeding well making an intracranial/neuro process less likely as well. Overall Dr Williams suggested that a likely diagnosis could be delayed maturation of the cns/respiratory axis causing bradypnea in sleep and associated desaturations, which may take 1-2 weeks to improve. She suggested trending a VBG tomorrow and a repeat CXR to ensure there is not ongoing CO2 retention in keeping with true pathologic hypoventilation. She also suggested considering adding a lactate and ammonia on to AM labs to complete a metabolic assessment.

 In summary our plan based on Dr Williams recommendations are as follows:

 1. VBG, lactate and ammonia tomorrow morning, with a repeat CXR ordered tomorrow as well. To assess for hypoventilation and reassess lung volumes on CXR
 2. Discontinue antibiotics tomorrow after completing a 5 day course. If new apneas consider full septic work up with LP and restarting Abx
 3. If persistent LFNC needs develop consider transitioning to HFNC/CPAP for added pressure and lung recruitment. At that point in time a transfer to McMaster for further assessment and Echo would be warranted.
 4. If no change in infant's status by next week consider re-contacting McMaster and consider transfer for Echo
 5. We will contact McMaster if the infants clinical status worsens as well
 6. Parents were updated at the bedside regarding the above conversation and questions were answered.
 7. We will also trial Prevacid 1mg/kg/d to see if treatment of GERD may help with desat/brady episodes.
""")

doc_list.append("""Document date: May 2, 2023
Subjective 
GA: 38w+2days, Mother G4T2P0L1, 40yo at birth.
 CGA: 39w
 BW: 3.335kg, CW: 3.150 (-5.5%)

 Sebastian is a is a 6-day old term infant born through SVD with APGAR 9 at 1 and 5 min who presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but she has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - Only had x2 desaturation episodes (no apnea/bradycardia) yesterday afternoon. No episodes overnight. x1 brief uncharted desaturation episode this AM to 80s which spontaneously resolved
 - Has not required O2, currently on room air
 - According to nursing, baby was overall uneventful overnight

 2. Fluid and nutrition
 - No struggles with feeding, currently breastfeeding adlib with formula top-up and on 2mL/hr D10 0.2%NS
 - Weight has been trending upwards (3.150 May 2 <- 3.135 April 30)
 - Has continued to have mild reflux, episodes this AM as per nurse
 - Trail of Prevacid for GERD, first dose given this morning


 3. Hyperbilirubinemia
 - Bilirubin still elevated at 246 but has been continually trending downwards
 
Assessment/Plan 
Sebastian is a 6-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Currently the differential remains open as we continue to investigate potential etiologies:

 1. Respiratory
 - Persistent episodes of desaturation have overall been improving but not completely resolved, O2 has not been needed
 - Last apneic episode on April 29 at 0400, apnea-free for >72 hours,
 - Last bradyardic episode on May 1 at 0630, bradycardia-free for >24hrs
 - As per MUMC consultation, infectious etiology is unlikely so abx can be stopped after 5 day course. But if clinical status acutely changes, reassessment would be appropriate
- CXR pending
 - MUMC also noted that these episodes may be due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve
 - VBGs show mild changes suggestive of hypoventilation (elevated CO2, low O2, low pH), lactate and NH3 normal, will repeat VBGs tomorrow AM
 - Overall, situation is improving, will continue to monitor and will consider re-consultation with MUMC if episodes do not resolve next week (May 8)

 2. Fluid/Nutrition
 - Feeding well but still has mild reflux that may be contributing to some episodes of desaturation
 - Started Prevacid trial this morning for continuing reflux
 - Plan to continue breastfeeding ad lib with formula top-up and D10-0.2%NS

 3. Hyperbilirubinemia
 - Bilirubin levels improving, no major risk factors
 - Will not need phototherapy 


""")
doc_list.append("""Document date: May 3, 2023
Subjective 
GA: 38w+2days
 BW: 3.335kg, CW: 3.190 (-4.3%)
 Mother G4T2P0L1, 40yo at birth.

 Sebastian is a is a 7-day old term infant born through SVD with APGAR 9 at 1 and 5 min who presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but she has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - Previously improving, but x3 destauration/bradycardia episodes overnight (x1 brady, x1 desat, x1 desat+brady) but no apneas
 - Episodes required gentle stimulation and O2

 2. Fluid and nutrition
 - No struggles with feeding, currently breastfeeding adlib with formula top-up and on 2mL/hr D10 0.2%NS
 - Weight has been steadily trending upwards (3.190 May 3 <- 3.150 May 2 <- 3.135 April 30)
 - Trail of Prevacid for GERD, first dose given yesterday
 - No reflux episodes this AM
 
Assessment/Plan 
Sebastian is a 7-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Top differential is delayed maturation of the cns/respiratory but diagnosis remains open as clinical status fluctuates.

 1. Respiratory
 - Persistent episodes of desaturation/bradycardia, x3 last night
 - Last apneic episode on April 29 at 0400, apnea-free for 4 days now
 - CXR yesterday returned normal
 - Repeat VBGs borderline with slightly elevated CO2 (44) but improved from yesterday
 - Full workup otherwise unremarkable for cardiac, respiratory, or infectious etiology
 - As per MUMC consultation, these episodes may be due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve
 - Overall, situation is stable, will continue to monitor and will consider re-consultation with MUMC if episodes do not resolve next week (May 8)
 - If situation deteriorates, will consult MUMC earlier

 2. Fluid/Nutrition
 - Currently breastfeeding ad lib with formula top-up and D10-0.2%NS
 - Started Prevacid trial yesterday for mild reflux
 - Hiccups (unchanged from previously) but no reflux this AM
 - Has been steadily gaining weight
 - Plan to continue current feeding regime


""")
doc_list.append("""Document date: May 4, 2023
Subjective 
GA: 38w+2days
 BW: 3.335kg, CW: 3.205 (-3.9%)
 Mother G4T2P0L1, 40yo at birth.

 Sebastian is a is a 8-day old term infant born through SVD with APGAR 9 at 1 and 5 min who presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but she has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - Clinically improving, x2 desat episodes yesterday: x1 desat only at ~1800 yesterday requiring gentle stimulation, x1 desat with bradycardia at ~2000 that self-resolved (was very brief so was not documented)
 - No apneas

 2. Fluid and nutrition
 - No struggles with feeding, currently breastfeeding adlib with formula top-up and on 2mL/hr D10 0.2%NS
 - Weight has been steadily trending upwards (3.205 May 4 <- 3.190 May 3 <- 3.150 May 2)
 - Trail of Prevacid for GERD, first dose given May 2
 - No spit-up/emesis episodes yesterday as per nursing team
 - Feeding well (60-70mL each session)

Assessment/Plan 
Sebastian is a 8-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Top differential is delayed maturation of the cns/respiratory but diagnosis remains open as clinical status fluctuates.

 1. Respiratory
 - Episodes improving, only x2 episodes yesterday
 - Last apneic episode on April 29 at 0400, apnea-free for 5 days now
 - Full workup otherwise unremarkable for cardiac, respiratory, or infectious etiology
 - As per MUMC consultation, these episodes may be due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve
 - Overall, situation is stable, will continue to monitor and will consider re-consultation with MUMC if episodes do not resolve next week (May 8)
 - If situation deteriorates, will consult MUMC earlier

 2. Fluid/Nutrition
 - Currently breastfeeding ad lib with formula top-up and D10-0.2%NS
 - Started Prevacid trial May 2 for mild reflux
 - Has not had any reflux yesterday
 - Has been steadily gaining weight
 - Plan to continue current feeding regime


""")
doc_list.append("""Document date: May 5, 2023
Subjective 
GA: 38w+2days
 BW: 3.335kg, CW: 3.260 (-2.2%)
 Mother G4T2P0L1, 40yo at birth.

 Sebastian is a is a 9-day old term infant born through SVD with APGAR 9 at 1 and 5 min who presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but she has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - Clinically improving, x2 desat/bradycardia episodes May 3-4, overnight today had 1-2 desaturation episodes (no bradycardia/apnea) that were very brief and self-resolving (not charted)

 2. Fluid and nutrition
 - No struggles with feeding, currently breastfeeding adlib with formula top-up and on 2mL/hr D10 0.2%NS
 - Weight has been steadily trending upwards (3.260 May 5 <- 3.205 May 4 <- 3.190 May 3)
 - Trail of Prevacid for GERD, first dose given May 2, no spit-up/emesis episodes since May 3
 - Feeding well (60-70mL each session)
 
Assessment/Plan 
Sebastian is a 9-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Top differential is delayed maturation of the cns/respiratory but diagnosis remains open as clinical status fluctuates.

 1. Respiratory
 - Episodes improving, only a few desaturation episodes last night that were brief and self-resolving
 - Last apneic episode on April 29 at 0400, apnea-free for 6 days now
 - As per MUMC consultation, episodes likely due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve
 - Can consider discharge after 5-7 days without any signficant events; today marks day #1 of being episode-free, May 10th would be earliest date to consider discharge if he does not have any additional significant episodes
 - Consider re-consultation with MUMC if episodes do not resolve next week (May 8) or if situation deteriorates


 2. Fluid/Nutrition
 - Currently breastfeeding ad lib with formula top-up and D10-0.2%NS
 - Started Prevacid trial May 2 for mild reflux and has not had any reflux since May 3
 - Previously discussed pros/cons of PPI with family, including increased risk of fractures as toddler due to effects on Ca absorption. Topic revisited today with family, plan to reassess PPI requirement in 1mo on outpatient basis but currently will keep baby on PPI as it seems to have helped their reflux
 - Steadily gaining weight
 - Continue current feeding regime

 Resolved issues
 1. Hyperbilirubinemia
 - Bilirubin levels improving, no major risk factors
 - Will not need phototherapy 

""")
doc_list.append("""Document date: May 6, 2023
Subjective 
GA: 38w+2days
 DOL 11
 BW: 3.335kg
 CW: 3.280 (gain 20 g)

 Mother G4T2P0L1, 40yo at birth.

 Sebastian presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but he has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - Previously had been improving with last apneic episode on May 4th, however overnight again had 2 episodes of apnea/bradycardia, these were self-resolving
  -past documented MUMC consultation felt that episodes likely due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve

 2. Fluid and nutrition
 - breastfeeding ad lib with formula top-up
 - On Prevacid for GERD, first dose given May 2, no spit-up/emesis episodes since May 3

 3. ID
 - remains afebrile
 
Assessment/Plan 
Term baby, remains admitted with ongoing apnea/desaturations thought to be secondary to delayed maturation of the cns/respiratory system. Ongoing self-resolving apnea/desaturations. Otherwise feeding well. Continue to monitor.
""")

doc_list.append("""Document date: May 7, 2023
Subjective 
GA: 38w+2days
 CGA 40w
 DOL 12
 BW: 3.335kg
 CW: 3.300 (gain 20 g)

 Mother G4T2P0L1, 40yo at birth.

 Sebastian presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but he has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - Had 3 bradycardia/desaturation episodes yesterday which were self-resolving, had previously been improving with no episodes for 2 days
  -past documented MUMC consultation felt that episodes likely due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve

 2. Fluid and nutrition
 - breastfeeding ad lib with formula top-up
 - On Prevacid for GERD, first dose given May 2, no spit-up/emesis episodes since May 3
 - Nursing team notes that he is quite gasey and irritable if he is awake, will try to pause after feeding to assess whether he is full before feeding more

 3. ID
 - remains afebrile

Assessment/Plan 
Sebastian is a 9-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Top differential is delayed maturation of the cns/respiratory but diagnosis remains open as clinical status fluctuates.

 1. Respiratory
 - Has had continued self-resolving bradycardia/desaturation episodes over the weekend, may consider consulting MUMC again tomorrow
 - Can consider discharge after 5-7 days without any signficant events; May 4th marks day #1 of being episode-free, May 10th would be earliest date to consider discharge if he does not have any additional significant episodes
 - Otherwise, stable on CPAP

 2. Fluid/Nutrition
 - Continue current feeding regime, try pausing after breastfeeding
""")

doc_list.append("""Document date: May 8, 2023
Subjective 
GA: 38w+2days
 BW: 3.335kg
 CW: 3.355kg
 Mother G4T2P0L1, 40yo at birth.

 Sebastian is a is a 12-day old term infant that presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but he has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - x2 periods of 3-8mins overnight where SpO2 was intermittently dipping +/- bradycardia, lowest SpO2 was ~79%
- Nurse notes that during these episodes, baby holds breath, then followed by desat and subsequent resumption of breathing,
 - All episodes were self-resolving 

 2. Fluid and nutrition
 - Feeding well via breastfeeding ad lib with formula top-up
 - On Prevacid for GERD, first dose given May 2, no spit-up/emesis episodes since May 3
 - Weight has been steadily trending upwards (3.355 May 7 <- 3.300 May 6 <- 3.280 May 5)

Assessment/Plan 
Sebastian is a 9-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Top differential is delayed maturation of the cns/respiratory but diagnosis remains open as clinical status fluctuates.

 1. Respiratory
 - Continued to have bradycardia/desaturation episodes over the weekend, all were self resolving but have not decreased in frequency
 - Last apneic episode on April 29 at 0400, apnea-free for 9 days now
 - As per MUMC consultation, episodes likely due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve, can consider discharge after 5-7 days without any signficant events
 - Unfortunately has continued to have episodes, will attempt re-consultation with MUMC today for possible further workup


 2. Fluid/Nutrition
 - Currently breastfeeding ad lib with formula top-up (D10-0.2%NS discontinued May 7)
 - Started Prevacid trial May 2 for mild reflux and has not had any reflux since May 3
 - Previously discussed pros/cons of PPI with family. Will reassess PPI requirement in 1mo on outpatient basis but currently will keep baby on PPI as it seems to have helped their reflux
 - Steadily gaining weight
 - Continue current feeding regime
""")

doc_list.append("""Document date: May 9, 2023
Subjective 
GA: 38w+2days
 BW: 3.335kg
 CW: 3.345kg
 Mother G4T2P0L1, 40yo at birth.

 Sebastian is a is a 13-day old term infant that presented with increased work of breathing at 7 minutes of life, requiring CPAP 6cmH2O and admission to NICU on April 25th PM. The working diagnosis at the time was TTN but he has continued to have episodes of desaturations +/- bradycardia/apnea.

 1. Respiratory
 - x1 period of desaturation (unknown duration), no need for stimulation
 - Nursing team notes that during these episodes, baby holds breath, then followed by desat and subsequent resumption of breathing,
 - All episodes were self-resolving 

 2. Fluid and nutrition
 - Feeding well via breastfeeding ad lib with formula top-up
 - On Prevacid for GERD, first dose given May 2, no spit-up/emesis episodes since May 3
 - Above birthweight, has been overall trending upwards (3.345 May 8 <- 3.355 May 7 <- 3.300 May 6)

Assessment/Plan
Sebastian is a 13-day old infant, who initially presented with tachypnea which was in keeping with TTN but is now experiencing persistent episodes of desaturation +/- bradycardia. Top differential is delayed maturation of the cns/respiratory but diagnosis remains open as clinical status fluctuates.

 1. Respiratory
 - As per MUMC consultation, episodes likely due to delayed maturation of the cns/respiratory, which may take 1-2 weeks to improve, can consider discharge after 5-7 days without any signficant events
 - Last notable apneic episode on April 29 at 0400, but unfortunately continues to have self-resolving desaturation +/- bradycardia episodes
 - Re-consulted with MUMC today and they agreed to patient transfer, will arrange transportation logistics and proceed with transfer as soon as ready

 2. Fluid/Nutrition
 - Currently breastfeeding ad lib with formula top-up (D10-0.2%NS discontinued May 7)
 - Started Prevacid trial May 2 for mild reflux and has not had any reflux since May 3
 - Previously discussed pros/cons of PPI with family. Will reassess PPI requirement in 1mo on outpatient basis but currently will keep baby on PPI as it seems to have helped their reflux
 - Steadily gaining weight
 - Continue current feeding regime

""")






#%% V1 - whole corpora processing 
def genQuery(doc_list):
    corpora = "\n\n".join(doc_list)
    query = F"{corpora}\n\nList all the abnormal findings mentioned in these notes and their dates:"
    query = F"{corpora}\n\nList all the treatments mentioned in these notes and their dates:"
    return query

doc_query = genQuery(doc_list=doc_list)

system_query = "You are a healthcare professional that understands medical jargon. Below is a collection of medical notes on a patient throughout their hospital stay. Each medical note is delimited by a 'Document date: <date of medical note>' header which also indicates which date the medical note was taken."

response = queryGPT(doc_query, system_query, model="gpt-4")


print(response["choices"][0]["message"]["content"])
#%% V2 - modular processing 


event_grps = """
1. Clinical events. For example:
	- Patient complaint or symptom
	- Physical exam finding
	- Any clinical assessment scores (e.g., CAM score, MoCA score, NIHSS score)
	- Improvement or worsening in clinical status 
	- Adverse event or complication (e.g., a fall)
2. Investigation results events. For example:
	- Vital signs
	- Lab results
	- Imaging results 
	- Pathology results
	- Microbiology results
	- Biopsy results
	- Any non-invasive test results (e.g., ECG, EEG) 
	- Any invasive test results (e.g., colonoscopy, EGD)
3. Patient care event. For example:
	- New or changes in diet orders (e.g., NPO, clear fluids only, renal diet)
	- New or changes in activity orders (e.g., bed rest, activity as tolerated)
	- New or changes in monitoring orders (e.g., vitals every 4hrs)
	- New or changes in wound care order (e.g., change wound dressing every day)
	- Any order for nursing care (e.g., keep head of bed elevated)
4.  Therapy event. For example:
	- New or changes in orders for oxygen therapy or ventilation 
	- New or changes in orders for ventilation (e.g., CPAP, BIPAP)
	- Intubation and extubation events
	- Care involving consumable supplies (e.g., casts, collars)
5. Medication event. For example:
	- New medication
	- Changes in medication
6. Procedural event. For example:
	- Bedside procedure (e.g., abscess drainage, IV insertion) 
	- Surgery
7. Logistical event. For example:
	- Communication between healthcare professionals (e.g., "Dr. Smith spoke to Dr. Lee")
	- Consult to another medical specialty 
	- Consult to allied health (e.g., occupational/physical therapy, pharmacy)
	- Transfer to another ward or facility 
"""
event_grps = """
1. Clinical events
2. Investigation results events
3. Patient care event
4. Therapy event
5. Medication event
6. Procedural event
7. Logistical event
"""

system_query = (
    "You are a healthcare professional that understands medical jargon and medical abbreviations. "
    "You will be given a medical note to extract medical events into a Markdown table. "
    "Below is an outline of the types of medical events that you are interested in:\n"
    F"{event_grps}"
    # "Below is a collection of medical notes on a patient throughout their hospital stay."
    # "Each medical note is delimited by a 'Document date: <date of medical note>'"
    # "header which also indicates which date the medical note was taken."
    )

def genQuery(doc):
    query = (
        "Below is a clinical document:\n\n"
        F"{doc}\n\n"
        "Extract all clinical events and occurrences from the above clinical document into a Markdown table. "
        "Each row should only have ONE clinical event (i.e., each clinical should be its own row. "
        "Deduce the event date using the document date as reference "
        "(e.g., if note mentions 'done yesterday' then the date of the event would be one day before the document date). "
        "Again, each clinical event should be added to another row. "
        "The table should have the following header: \n"
        "| Medical problem that event is related to | "
        "Clinical event | "
        "Event date (or 'N/A' if not mentioned) | "
        "Reasoning for 'Event date' output |\n"
        "| ------- | ------- | ------- | ------- |"
        # "List all of the medical events in the above medical document, "
        # "grouped according to the above event groupings in a Markdown table. "
        # "If a group does not have any events, report 'None mentioned'. "
        # "Here is the header of the Markdown table:\n"
        # "| Event type (Clinical, ) | Investigation events | Patient care events | Medication events | Procedural events |\n"
        # "| --- | --- | --- | --- | --- |"
    )
    return query

responses_list: list[GPTResponse] = []
print(system_query)

for i in range(4):
    doc_query = genQuery(doc_list[i])
    match = re.search(R"Document date: ([^\n]+)\n", doc_list[0], re.M)
    print(doc_query)

    doc_date = match.group(1)
    response = queryGPT(doc_query, system_query)
    gpt_response = GPTResponse(doc_date,
                               response["choices"][0]["message"]["content"],
                               response)
    responses_list.append(gpt_response)
    print(gpt_response.string)

responses_raw_list: list[str] = [r.string for r in responses_list]


#%%
responses_df = pd.DataFrame()
for gpt_response in responses_list:
    print(gpt_response.string)
    df = pd.read_csv(StringIO(gpt_response.string), sep="|")
    df = df.iloc[1:] # Skip first column which is the division border
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Remove unnamed columns
    df.columns = df.columns.str.strip() # Strip whitespace from column name ends
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x) # Strip spaces from strings
    df[COL_DATE] = df[COL_DATE].str.replace("N/A", F"({gpt_response.date})")
    responses_df = pd.concat([responses_df, df])

responses_df = responses_df.drop([COL_REASON], axis=1) # Drop reasoning to compress data
responses_df.sort_values(COL_MEDPROB)
#%%
responses_md = responses_df.to_markdown(index=False)
print(responses_md)

#%%
match = re.search(R"Document date: ([^\n]+)\n", doc_list[0], re.M)
match.group(1)