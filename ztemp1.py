#%%
import re, time, json, os

import openai
from private.keys import KEY1
openai.api_key = KEY1

import pandas as pd
#%%


def genQuery(doc_list):
    chars = "\n".join([F"- {i}" for i in doc_list])
    md_table = '| ' + ' | '.join(['Characteristic (verbatim from above)',
                           'Does the paper contain this characteristic (only respond with "Yes", "No", or "Not mentioned")',
                           'If "Yes", which sentence(s) in the abstract contain this info. If "No" or "Not mentioned", give reason'
                           ]) + ' |'
    suffix = F'Report if the abstract above contains the characteristics listed above in the following markdown table using only "Yes", "No", or "Not mentioned":\n{md_table}'
    query = F"Abstract:\n\n{abstract}\n\nCharacteristics:\n{chars}\n\n{suffix}"
    return query

system_query = "You are a healthcare professional that understands medical jargon. You will be given medical documents containing information pertaining to a patient to summarize"

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
        output_raw = queryGPT(query=query, model=model)
        
    return output_raw

#%% Test runs
doc1 = """
Document date: March 30, 2023
ED PROVIDER NOTES
Michael Stephen Sandy is a 52 y.o. male who was seen in the ED with Altered Level of Consciousness

The patient has a history of a form of neuro logical disorder (unknown at this time).  He also has a known cerebral aneurysm of 8.2 mm.  He was admitted to the Hamilton General in September last year.  Unfortunately I do not see any notes on our system at this time.
 
He was at home in his usual state of health.  He had gone to the washroom.  His grandson then heard him fall.  He was found to be prone on the bathroom floor with some facial trauma secondary to the glasses.  He was unresponsive.  EMS was called.
 
EMS found the patient to be hypotensive with systolics in the 80s.  GCS 4.  Glasses embedded somewhat in the facial skin of the nose.  Heart rate 90s.  Blood glucose normal.  There was a concern with EMS ECG of ST elevations but these were only in aVL and there were ST depressions in other leads.

Initial Impression and Plan:
The presentation is very concerning for intracranial hemorrhage secondary subarachnoid bleed.  Initially the concern was for hypotension which we managed with repeated doses of phenylephrine and Levophed infusion was started.  Following an improvement in blood pressure from 70 systolic up to over 120 systolic, we proceeded to intubate the patient.  He had been bagged with an oral airway up to this time.
 
Intubation was performed by the respiratory therapist using glide scope.  Grade 3 view was obtained, it was somewhat difficult to pass the tube but this was done with only transient hypo hypoxia to the mid 80s.  The tube was secured in place and confirmed with qualitative capnometry as well as bilateral breath sounds.  Hemodynamic status stable per intubation mind you the patient was on Levophed infusion at 0.1 mic per kilogram per minute at this time.
 
Portal chest x-ray shows good endotracheal tube position as well as OG tube position.  The patient was taken to CT immediately following this.
 
I got a call from radiology at 10:20 AM stating that there is signs of subarachnoid hemorrhage.  I have ordered a CT angio as the patient is still in the radiology suite.  I have completed emergency consent which has been scanned in at this time.
 
I really updated the patient's wife prior to the patient going to CT on the likelihood of an intracranial hemorrhage and the patient's current critical condition.
 
At this time (10:30 AM) I paged neurosurgery and awaiting their call back.  The patient is still in CT obtaining a CT angiogram.
 
(Time spent in the direct care of the patient at this time was approximately 55 minutes)

"""

doc2 = """
Document date: March 30, 2023
NEUROSURGERY CONSULT NOTE
History of Presenting Illness:
Michael Stephen Sandy is a 52 y.o. male presenting with Ruptured cerebral aneurysm.
 
HPI 
Wife reports he fell down while on the toilet and became unconscious. No movement since then. As per ER and EMS he was GCS 4 and not moving on presentation. Proceeded with intubation and CT followed by angio after seeing SAH.
 
Wife, Dawn, reports that he has a known 8.2mm aneurysm that was diagnosed roughly 3 months ago.

Assessment & Plan
 
Principal Problem:
  Ruptured cerebral aneurysm
 
This is a 52 y.o. year old male with subarachnoid hemorrhage due to a Anterior Communicating Artery (ACoA) Aneurysm. This appears to be a saccular aneurysm. The neurointerventional team will be consulted. 
 
Aneurysm grade is:
WFNS Grade 5: GCS of 3-6. Historical survival of grade 5 patients is 10%. 
Hunt & Hess Grade 5: Deep coma, decerebrate rigidity, moribund. This suggests a 70-77% early mortality risk. 
Modified Fisher Score 3: Dense SAH with no IVH. This suggests a 33% risk of symptomatic vasospasm.
 
Management includes:
Admission to ICU with close monitoring. The patient will be placed on Nimodipine 60mg PO q4h for 21 days improved functional outcome in SAH. Systolic blood pressure will be maintained at 100-140 while the aneurysm remains unsecured. Definitive management will be discussed with the INR team. The patient will be monitored for common complications including rebleeding, hydrocephalus, vasospasm, stress induced cardiomyopathy, arrhythmia, SIADH, and seizure. The patient will have moon boots for DVT prophylaxis. 
"""

doc3 = """
Document date: March 30, 2023
ICU EAST ADMISSION NOTE
 
ID: Micheal Stephen Sandy is a 52 year old male with facial trauma and altered level of consciousness secondary to fall and ruptured cerebral aneurysm
 
HPI: 
Michael is a 52 year old male with a known cerebral aneurysm found incidentally on ED visit in August 2022 for unrelated falls secondary to neuropathy syndrome (later thought to be small fiber neuropathy). He was at home today and went to the washroom when his grandson heard him fall. His grandson found him lying prone and unresponsive with facial trauma secondary to his glasses. 
 
EMS was called and in the ED Michael was found to be hypotensive and GCS 4; after improving his BP, the patient was intubated and after CT studies he was brought to the ICU. On CT he was found to have ruptured his cerebral aneurysm leading to extensive subarachnoid hemorrhage. Neurosurgery is aware and will see. 
"""

doc4 = """
Neurosurgery Operative Report
30/3/2023
Start time: 1205
 
Procedure: right placement of external ventricular drain
 
Surgeons: Dr Algird
Dr Alkhoori R1
Anesthesia: local anesthetic at incision and general
 
Estimated Blood Loss: Minimal
 
Complications: None
 
Passes: 1
 
Initial ICP: 4
 
Clinical Note:
Michael Stephen Sandy is a 52 y.o. male known to have Ruptured cerebral aneurysm. An external ventricular drain was indicated for intraventricular hemorrhage ICP monitoring. This was performed as an emergent procedure. 
Consent was discussed with the SDM prior to the procedure. Risks included but are not limited to infection, hemorrhage, neurologic compromise or disability, seizure, stroke, CSF leak, recurrence, cardiorespiratory complications, anesthetic risks, and death. The SDM understands the indications and benefits for this procedure. 
 
Operative Details:
This procedure was performed at ICU. Imaging was made available for reference throughout the procedure. Bloodwork was referenced prior to ensure no coagulopathy. The patient received Cefazolin 2g IV prophylaxis. A brief pre-operative check was performed. In the supine position, the patient was prepped and draped in the usual sterile fashion. The appropriate craniostomy site at Kocher's point and intended tunneling site 5cm away were measured and infiltrated with local anesthetic. An initial incision was made down to bone at the craniostomy site. Hemostasis was ensured throughout with pressure. Craniostomy was drilled perpendicular to the bone, on a trajectory toward the lateral ventricle at the intersection of the medial canthus and tragus. Dura was punctured. The EVD was inserted 7cm and spontaneous CSF flowed with an approximate pressure of 4cm. The CSF appeared blood tinted. The drain was then tunneled 5cm away. The EVD was secured with 2.0 prolene. The incisions were closed with 2.0 prolene in interrupted fashion. The operative site was covered with a tegaderm dressing. The ICP monitor was connected to the external apparatus. 
 
The patient tolerated the procedure well. There were no acute complications. Postoperatively, we will seek a CT scan to assess drain placement, ventricle size, and to rule out any acute complications. 
"""

doc5 = """
CARDIOLOGY CONSULTATION
31/03/23
 
IDENTIFICATION
·	Michael Stephen Sandy is a 52 y.o. male with extensive subarachnoid hemorrhage secondary to an Anterior Communicating Artery Aneurysm.
·	We were consulted regarding his troponin elevation.
HISTORY OF PRESENTING ILLNESS
Chief Complaint	Troponin Elevation
·	Michael Stephen Sandy is a 52 y.o. male with an unwitnessed fall on March 30, 2023.  He was found prone and unresponsive by his grandson.  Upon arrival to the Emergency Department, he was hypotensive.  He was subsequently intubated for a low GCS level.
·	He has had minimal neurological activity since his admission.
·	We were unable to elicit a symptom history regarding baseline chest discomfort or shortness of breath at rest or with exertion.

IMPRESSION
·	Michael Stephen Sandy is a 52 y.o. male with troponin elevation in the context of extensive subarachnoid hemorrhage.  There is no evidence of ST segment changes consistent with acute occlusive ischemia.  Given the clinical context, this is most likely due to myocardial injury from a stress-mediated process.  Furthermore, given the extensive subarachnoid hemorrhage, the harms of percutaneous coronary intervention would likely outweigh the benefits even if Michael were to develop an ST-elevation MI.
 
----------------------------------------
 
PLAN
We will continue to follow and have addressed the following issues:
·	Troponin elevation: We recommend ordering an echocardiogram.  As mentioned above, there is maximal harm and minimal benefit in treating Michael for a suspected acute coronary syndrome at this time.  As a result, we do not recommend initiating antithrombotic therapy.  Statin therapy may be reasonable.  We will order a lipid panel and hemoglobin A1C for the morning.

"""
doc6 = """
Neurosurgery Progress Note
31/3/2023
Michael Stephen Sandy is a 52 y.o. male with Ruptured cerebral aneurysm
 LOS: 1 day      
 
 
 
Subjective 
 
started to localise overnight
ICPs stable and nights of 14
EVD low output blood tinged 3-11 hourly
Assessment & Plan
 

Principal Problem:
  Ruptured cerebral aneurysm
Active Problems:
  Subarachnoid hemorrhage
 
Michael Stephen Sandy is a 52 y.o. male with Ruptured cerebral aneurysm POST 
EVD insertion
Plan
- INR consult 
"""
doc7 = """

"""

doc_list = [
    doc1,
    doc2,
    doc3,
    doc4,
    doc5,
    doc6,
    doc7,
    ]



#%%
print(queryGPT("Test run", system_query))

#%% Single run

abstract = "OBJECTIVE: The biomarkers glial fibrillary acid protein (GFAP) and S100B are increasingly used as prognostic tools in severe traumatic brain injury (TBI). Data for mild TBI are scarce. This study aims to analyze the predictive value of GFAP and S100B for outcome in mild TBI and the relation with imaging., METHODS: In 94 patients biomarkers were determined directly after admission. Collected data included injury severity, patient characteristics, admission CT, and MRI 3 months postinjury. Six months postinjury outcome was determined with Glasgow Outcome Scale Extended (GOSE) and return to work (RTW)., RESULTS: Mean GFAP was 0.25 mug/L (SD 1.08) and S100B 0.54 mug/L (SD 1.18). In 63% GFAP was not discernible. GFAP was increased in patients with an abnormal CT (1.20 mug/L, SD 2.65) compared to normal CT (0.05 mug/L, SD 0.17, p < 0.05). Also in patients with axonal injury on MRI GFAP was higher (0.65 mug/L, SD 0.91 vs 0.07 mug/L, SD 0.2, p < 0.05). GFAP was increased in patients with incomplete RTW compared to complete RTW (0.69 mug/L, SD 2.11 vs 0.12 mug/L, SD 0.38, p < 0.05). S100B was not related to outcome or imaging studies. In multivariate analysis GFAP was not predictive for outcome determined by GOSE and RTW., CONCLUSIONS: A relation between GFAP with imaging studies and outcome (determined by RTW) was found in contrast to S100B. As the positive predictive value of GFAP is limited in this category of TBI patients, this biomarker is not suitable for prediction of individual patient outcome."
chars_list = ["Uses human patients",
              "Patient population includes traumatic brain injury (TBI)",
              "Studies the relationship between a prognostic factor and the clinical outcomes of patients",
              "The study is not a review paper looking at other studies",
              "Is not a case study with only a few patients"
              ]

query = genQuery(chars_list, abstract)
output_raw = queryGPT(query)
main_response = output_raw["choices"][0]["message"]["content"]


stmts: list[str] = [item.strip() for item in main_response.split("\n")] # Split by newline, only include lines that have word characters
ledger = list(filter(lambda item: re.search(r"\w", item) == None, stmts))[0]
stmts_real = stmts[stmts.index(ledger) + 1 : ] # Get statements after ledger
stmts_items = [[i.strip() for i in filter(None, stm.split("|"))] for stm in stmts_real] # Filter with none to get rid of empty strings, should only return 3 items corresponding to the 3 columns of output
stmts_vec1 = [i[1] == "Yes" for i in stmts_items]

print(stmts_items)
print(stmts_vec1)
