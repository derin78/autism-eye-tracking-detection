# Dedication

We would like to dedicate this project first and foremost to our families, especially our parents, who have always believed in us and supported us through every step of our education. Without their patience and encouragement, none of this would have been possible.

We also dedicate this work to our supervisor, **[Supervisor Name]**, whose guidance, feedback, and constant support helped us stay on the right track from the very beginning until the end of this project.

A special dedication goes to all the children and families affected by Autism Spectrum Disorder. This project was built with them in mind, and we hope it can contribute, even in a small way, to making early screening more accessible.

Finally, we dedicate this research to all the teachers, staff, and fellow students at the **University of Sulaimani** who have been a part of our learning journey. Thank you for everything.

---

# Acknowledgments

## Option 1 (Simple)

First, we thank God for helping us complete this project.

We put in a lot of effort to get this project done on time. However, it would not have been possible without the support of many people. We are very grateful to our supervisor, **[Supervisor Name]**, for his guidance and supervision throughout the whole process, and for providing us with the information we needed to complete this work.

We would also like to thank our parents and friends who helped us a lot in finishing this project within the limited time we had.

**[Team Member Name 1]**
**[Team Member Name 2]**
**[Team Member Name 3]**
**[Team Member Name 4]**
**[Date]**

---

## Option 2 (Detailed)

First of all, we thank God for giving us the strength, patience, and ability to complete this project. It was not an easy journey, but we are grateful for every moment of it.

We worked hard to finish this project on time, but honestly, it would not have been possible without the help and support of many people around us. We are truly thankful to our supervisor, **[Supervisor Name]**, for his guidance and constant supervision throughout the entire process. He provided us with the direction and knowledge we needed, and his support made a real difference in bringing this project to life.

We also want to thank our parents for always being there for us, not just during this project but throughout our whole education. Their love and encouragement kept us going even when things got difficult. And to our friends who helped us along the way, whether it was advice, motivation, or just being there when we needed them — thank you.

Working on this project taught us so much, from building real software to understanding how eye tracking technology can be used in healthcare. It has been a valuable learning experience that we will carry with us.

**[Team Member Name 1]**
**[Team Member Name 2]**
**[Team Member Name 3]**
**[Team Member Name 4]**
**[Date]**

---

# Abstract

Autism Spectrum Disorder (ASD) is a developmental condition that affects how a person communicates and interacts with others. Early detection of ASD is very important because it allows children to receive support and therapy at a young age, which can greatly improve their development. However, most current screening methods rely on expensive eye tracking hardware that is not available in every clinic or school.

In this project, we developed a desktop application that uses a standard webcam to track eye movements and screen for early signs of ASD. The system is built with Python and uses MediaPipe Face Mesh to detect and track the user's iris position in real time without the need for any special equipment. The screening test is based on the Preferential Looking Paradigm, a well-known clinical method where the subject is shown split-screen images containing faces on one side and objects on the other. Research shows that children with ASD tend to look less at faces compared to typically developing children.

The gaze data collected during the test is analyzed using a Random Forest machine learning classifier trained on synthetic data generated from published clinical thresholds. The model achieved a cross-validation accuracy of 94.75%. After each test, the system automatically generates a report with the subject's risk level and detailed gaze statistics.

This project provides a low-cost, accessible, and easy-to-use tool that could help in the early screening of ASD using only a regular computer and webcam.
