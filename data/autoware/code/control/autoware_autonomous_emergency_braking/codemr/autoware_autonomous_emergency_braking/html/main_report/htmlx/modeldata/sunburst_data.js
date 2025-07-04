function EQ_GET_DATA(){ 
	  var ret = {
"name": " autoware_autonomous_emergency_braking", "value":138, 
"prmetrics":{"5":1,"6":1,"7":1,"8":1,"9":1,"10":1,"11":1},
"prmetricvalues":{"5":0,"6":3,"7":0,"8":138,"9":0,"10":1,"11":0},
"children": [ {
"name": "autonomous_emergency_braking", "value":138, 
"pmetrics":{"4":1,"12":1,"13":1,"14":1,"3":1,"1":1,"0":1,"6":1,"8":1,"2":1,"15":1,"16":1},
"pmetricvalues":{"4":1,"12":0,"13":0,"14":3,"3":1,"1":1,"0":1,"17":0.0,"6":3,"18":0.0,"8":138,"19":1.0,"2":1,"15":0,"16":23},
"children":[
{
"name": "CollisionDataKeeper","key": "m","value":24, 
"metrics":{"20":1,"21":1,"22":1,"23":1,"24":1,"25":1,"26":1,"0":1,"27":1,"28":1,"29":5,"30":1,"8":1,"31":1,"32":1,"33":1,"34":1,"35":1,"36":1,"16":1,"4":1,"2":1,"3":1,"1":1},
"metricvalues":{"20":1,"21":0,"22":2,"23":0.0,"24":22,"25":0,"26":0,"0":1,"27":7,"28":7,"29":1.0,"30":0.0,"8":24,"31":0,"32":0.524,"33":1,"34":0,"35":0,"36":0,"16":7,"4":1,"2":1,"3":1,"1":1}
},
{
"name": "ObjectData","key": "l","value":3, 
"metrics":{"20":1,"21":1,"22":1,"23":1,"24":1,"25":1,"26":1,"0":1,"27":1,"28":1,"29":1,"30":1,"8":1,"31":1,"32":1,"33":1,"34":1,"35":1,"36":1,"16":1,"4":1,"2":1,"3":1,"1":1},
"metricvalues":{"20":0,"21":0,"22":3,"23":0.0,"24":0,"25":0,"26":0,"0":1,"27":0,"28":0,"29":0.0,"30":0.0,"8":3,"31":0,"32":0.0,"33":0,"34":0,"35":0,"36":0,"16":0,"4":1,"2":1,"3":1,"1":1}
},
{
"name": "AEB","key": "n","value":111, 
"metrics":{"20":1,"21":1,"22":2,"23":1,"24":2,"25":1,"26":1,"0":1,"27":1,"28":1,"29":5,"30":1,"8":2,"31":1,"32":1,"33":1,"34":1,"35":1,"36":1,"16":1,"4":2,"2":1,"3":1,"1":1},
"metricvalues":{"20":2,"21":0,"22":30,"23":0.0,"24":81,"25":0,"26":0,"0":1,"27":3,"28":4,"29":1.0,"30":0.0,"8":111,"31":0,"32":0.444,"33":2,"34":1,"35":0,"36":0,"16":16,"4":2,"2":1,"3":1,"1":1}
}
]
}]
 }
;
return ret;
}
var EQ_METRIC_MAP = {};
EQ_METRIC_MAP["C3"] =0;
EQ_METRIC_MAP["Complexity"] =1;
EQ_METRIC_MAP["Coupling"] =2;
EQ_METRIC_MAP["Lack of Cohesion"] =3;
EQ_METRIC_MAP["Size"] =4;
EQ_METRIC_MAP["Number of Highly Problematic Classes"] =5;
EQ_METRIC_MAP["Number of Entities"] =6;
EQ_METRIC_MAP["Number of Problematic Classes"] =7;
EQ_METRIC_MAP["Class Lines of Code"] =8;
EQ_METRIC_MAP["Number of External Packages"] =9;
EQ_METRIC_MAP["Number of Packages"] =10;
EQ_METRIC_MAP["Number of External Entities"] =11;
EQ_METRIC_MAP["Efferent Coupling"] =12;
EQ_METRIC_MAP["Number of Interfaces"] =13;
EQ_METRIC_MAP["Number of Classes"] =14;
EQ_METRIC_MAP["Afferent Coupling"] =15;
EQ_METRIC_MAP["Weighted Method Count"] =16;
EQ_METRIC_MAP["Normalized Distance"] =17;
EQ_METRIC_MAP["Abstractness"] =18;
EQ_METRIC_MAP["Instability"] =19;
EQ_METRIC_MAP["Coupling Between Object Classes"] =20;
EQ_METRIC_MAP["Access to Foreign Data"] =21;
EQ_METRIC_MAP["Number of Fields"] =22;
EQ_METRIC_MAP["Specialization Index"] =23;
EQ_METRIC_MAP["Class-Methods Lines of Code"] =24;
EQ_METRIC_MAP["Number of Children"] =25;
EQ_METRIC_MAP["Depth of Inheritance Tree"] =26;
EQ_METRIC_MAP["Number of Methods"] =27;
EQ_METRIC_MAP["Response For a Class"] =28;
EQ_METRIC_MAP["Lack of Tight Class Cohesion"] =29;
EQ_METRIC_MAP["Lack of Cohesion of Methods"] =30;
EQ_METRIC_MAP["Number of Static Fields"] =31;
EQ_METRIC_MAP["Lack of Cohesion Among Methods(1-CAM)"] =32;
EQ_METRIC_MAP["CBO App"] =33;
EQ_METRIC_MAP["Simple Response For a Class"] =34;
EQ_METRIC_MAP["Number of Static Methods"] =35;
EQ_METRIC_MAP["CBO Lib"] =36;
var EQ_SELECTED_CLASS_METRIC 		= "C3";
var EQ_SELECTED_PACKAGE_METRIC 	= "C3";
var EQ_SELECTED_PROJECT_METRIC 	= "Class Lines of Code";
var EQ_CLASS_METRIC_INDEX 	= EQ_METRIC_MAP[EQ_SELECTED_CLASS_METRIC];
var EQ_PACKAGE_METRIC_INDEX	= EQ_METRIC_MAP[EQ_SELECTED_PACKAGE_METRIC];
var EQ_PROJECT_METRIC_INDEX 	= EQ_METRIC_MAP[EQ_SELECTED_PROJECT_METRIC];
var EQ_COLOR_OF_LEVELS = ["#1F77B4","#007F24","#62BF18","#FFC800","#FF5B13","#E50000"];
var EQ_CLASS_METRICS = ["C3","Complexity","Coupling","Lack of Cohesion","Size","Class Lines of Code","Weighted Method Count","Coupling Between Object Classes","Access to Foreign Data","Number of Fields","Specialization Index","Class-Methods Lines of Code","Number of Children","Depth of Inheritance Tree","Number of Methods","Response For a Class","Lack of Tight Class Cohesion","Lack of Cohesion of Methods","Number of Static Fields","Lack of Cohesion Among Methods(1-CAM)","CBO App","Simple Response For a Class","Number of Static Methods","CBO Lib"];
var EQ_PACKAGE_METRICS = ["C3","Complexity","Coupling","Lack of Cohesion","Size","Number of Entities","Class Lines of Code","Efferent Coupling","Number of Interfaces","Number of Classes","Afferent Coupling","Weighted Method Count","Normalized Distance","Abstractness","Instability"];
var EQ_PROJECT_METRICS = ["Number of Highly Problematic Classes","Number of Entities","Number of Problematic Classes","Class Lines of Code","Number of External Packages","Number of Packages","Number of External Entities"];
function EQ_GET_COLOR(d) {
if(d.metrics)
return EQ_COLOR_OF_LEVELS[d.metrics[EQ_CLASS_METRIC_INDEX]];
if(d.pmetrics)
return EQ_COLOR_OF_LEVELS[d.pmetrics[EQ_PACKAGE_METRIC_INDEX]];
if(d.prmetrics)
return EQ_COLOR_OF_LEVELS[d.prmetrics[EQ_PROJECT_METRIC_INDEX]];
return EQ_COLOR_OF_LEVELS[0];
}
