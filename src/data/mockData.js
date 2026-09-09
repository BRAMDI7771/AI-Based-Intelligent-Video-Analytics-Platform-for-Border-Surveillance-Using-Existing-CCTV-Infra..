export const cameras=[
{id:'C-07',kind:'Person',confidence:94,time:'10:08:31 AM'},
{id:'C-03',kind:'Vehicle',confidence:91,time:'10:07:42 AM'},
{id:'C-11',kind:'Person',confidence:96,time:'10:06:19 AM'},
{id:'C-02',kind:'Person',confidence:89,time:'10:05:33 AM'},
{id:'C-09',kind:'Vehicle',confidence:93,time:'10:04:18 AM'},
{id:'C-05',kind:'Person',confidence:87,time:'10:03:51 AM'}
];
export const alerts=[
{id:1,type:'HUMAN INTRUSION',camera:'C-07',confidence:94.8,level:'CRITICAL',time:'10:08:31 AM'},
{id:2,type:'SUSPICIOUS ACTIVITY',camera:'C-01',confidence:87.3,level:'HIGH',time:'10:07:12 AM'},
{id:3,type:'VEHICLE DETECTED',camera:'C-03',confidence:92.1,level:'MEDIUM',time:'10:06:19 AM'},
{id:4,type:'PERSON DETECTED',camera:'C-04',confidence:88.6,level:'INFO',time:'10:05:33 AM'}
];
export const persons=[{camera:'C-07',status:'Moving'},{camera:'C-04',status:'Stationary'},{camera:'C-09',status:'Moving'},{camera:'C-02',status:'Moving'},{camera:'C-11',status:'Moving'}];
export const vehicles=[{type:'Car',plate:'UP32 XX 1234',time:'10:07:42 AM'},{type:'Bike',plate:'UP32 AB 7812',time:'10:06:19 AM'},{type:'Bike',plate:'UP32 CD 4521',time:'10:05:33 AM'},{type:'Car',plate:'UP32 PQ 9087',time:'10:04:18 AM'},{type:'Bike',plate:'UP32 KL 6412',time:'10:03:51 AM'}];
export const events=[{time:'10:08:31 AM',event:'Human Intrusion',camera:'C-07',level:'CRITICAL'},{time:'10:07:42 AM',event:'Vehicle Detected',camera:'C-03',level:'MEDIUM'},{time:'10:06:19 AM',event:'Person Detected',camera:'C-04',level:'HIGH'},{time:'10:05:33 AM',event:'Suspicious Activity',camera:'C-09',level:'HIGH'},{time:'10:04:18 AM',event:'Vehicle Detected',camera:'C-11',level:'MEDIUM'}];
