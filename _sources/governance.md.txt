# Governance

The accelerator simulation environment SciBmad is led in an open governance model, described in this file.

## Steering Committee

### Current Roster

- Joseph Devlin
- Georg Hoffstaetter de Torquat
- David Sagan
- Matt Signorelli

### Role

Members of the steering committee (SC) can change GitHub access settings, do administrative operations such as rename/move/archive repositories, change branch protection rules, etc. Each change will immediately be communicated to the Steering Committee, so each member has a chance to voice objections.
SC members can call votes for decisions (technical or governance).
Decisions are documented at https://docs.google.com/document/d/1r18-ktpJ_DkZra0tUvd0lZSvEn_aQDmD6LB7K2gasJI
The SC can change the governance structure, but only in a unanimous vote.

### Decision Process

Decision of the SC usually happen in the weekly SC meeting (initially Tuesdays at 4pm EST).
The weekly SC meeting is by invitation only, individual SC members can invite to the SC meeting.
Decisions are made in a non-confidential manner, by majority on the cast votes of SC members.
In tie situations, decisions will be postponed until an acceptable solution is found.
Votes in the SC are cast in a non-confidential manner.

### Appointment Process

New members are appointed to the SC by the current SC members by unanimous vote. As a SC member, regularly attending and contributing to the SC meetings is expected. 
SC members can resign or be removed by majority vote, e.g., due to inactivity, bad acting, or other reasons.

## Technical Committee

The people most involved in the programming of SciBmad are trusted to act in the full code’s best interest and TC membership should enable them to develop the code without much interference.

### Current Roster

- Joe Devlin
- David Sagan
- Matt Signorelli

### Role

Individual TC members can accept pull requests from all TC members.
Pull requests from non-TC members need two approvals to be accepted. This number can be changed when the number of TC members increases and should remain at least 50% of the number of TC members.
The TC can propose governance changes to the SC.
The TC can propose new TC members to be approved by the SC.

### Decision Process

Discussion in the TC usually happens in the weekly developer meetings (currently Thursday 4pm EST).
The weekly developer meeting is open to the public.
Votes in the TC are cast in a non-confidential manner.
Meeting notes and decisions of the weekly developers meeting are being maintained at https://docs.google.com/document/d/1PLLP6VJxmtq8eQGjdXAwLCNX66hAswFtFeLSL3kq2PQ/edit?tab=t.0

### Appointment Process

Members of the TC are appointed by the steering committee by majority vote.
Any TC or SC member can suggest new TC members for a vote of the SC.
TC members are expected to regularly attend and contribute to the weekly developer meetings.
SC members can also be TC members.

TC members can resign or be removed by majority vote by either TC or SC. An explanation does not have to be publicly distributed.

## Former Members

The meeting notes of the weekly governance meeting will track former memberships of the SC.
The meeting notes of the weekly developers meeting will track former memberships of the TC.

## AI Contribution Guidelines

Contributors may use AI tools, such as Large Language Models (LLMs) and code assistants, when contributing to SciBmad. For SciBmad packages specified below as "Core Functionality Packages", the contributor is responsible for the quality of the result and for understanding what they submit. In particular, the guidelines for pull requests made to Core Functionality Packages are:


* **You must understand your contribution.** Do not submit code, documentation, or other content that you do not fully understand. You should be able to explain to reviewers what your contribution does, how it works, and how you verified it. You should be able to answer reviewers' questions yourself, without recourse to AI. In particular, do not waste the time of other contributors by being a proxy between reviewers and an AI.
* **Review all AI-generated content.** Before submitting, verify that the contribution is correct, tested, and follows the project's contributing guidelines.
* **You own what you submit.** As the PR author, you are responsible for all changes, regardless of the tools used to create them, and must certify that you wrote or otherwise have the right to submit the code you are contributing by signing your commits.
* **Do not instruct AI tools to directly respond to comments.** When maintainers, component owners, or other contributors provide feedback on your PR, you may use AI tools to help craft responses, but you must fully understand and take ownership of what you write. Reviewers want to engage with you, not receive generic AI-generated responses. If you use AI assistance to formulate responses, ensure they genuinely reflect your understanding and that you can defend your reasoning.
* **Avoid verbose AI-generated content.** Trim down unnecessary verbosity in PR descriptions, commit messages, and code comments. Be clear, concise, and specific. Respect the time of maintainers and reviewers.

### Core Functionality Packages:
- SciBmad.jl
- BeamTracking.jl
- Beamlines.jl
- NonlinearNormalForm.jl
- GTPSA.jl
- AtomicAndPhysicalConstants.jl
- FundamentalFrequencies.jl
