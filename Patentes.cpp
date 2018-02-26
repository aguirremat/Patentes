//#include "/home4/matias/proyecto/opencv-2.4.13.4/release/opencv_contrib/modules/text/include/opencv2/text/erfilter.hpp"
//#include "/home/matias/proyecto/opencv-2.4.13.4/release/opencv_contrib/modules/text/include/opencv2/text/textDetector.hpp"
//#include "/home/matias/proyecto/opencv-2.4.13.4/release/opencv_contrib/modules/text/include/opencv2/text/ocr.hpp"
//#include "/home/matias/proyecto/opencv-2.4.13.4/release/opencv_contrib/modules/text/include/opencv2/text.hpp"
#include "opencv2/text.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ioctl.h>


//#include "tesseractclass.cpp"


#define MUESTRAS 5

using namespace std;
using namespace cv;
using namespace cv::text;


//	funciones   	 //

void Filtros   		(int ,void*);
void Recorte_ancho	(Mat , int );
void Recorte_alto	(Mat , int );
void Lectura		(int ,void*);
void T_morfologica	(Mat , int );


//	Global variables //

Mat src[MUESTRAS],src_gray[MUESTRAS];
Mat Guardado[MUESTRAS];
Mat dst, detected_edges;
//char nombre[40]


int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
int  Minimo_Filtro=0;
int  Maximo_Filtro=10;
double rho=10;
double theta=1.57079;

//*Variables para umbralizacion*//
int threshold_value = 0;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
 
//*Variables para funciones morfológicas*//
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

//Variables para recortar imagen//
Mat src_recortada_1[MUESTRAS];
Mat src_recortada_2[MUESTRAS];
RNG rng(12345);
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;




/*****************************************************************************************************

					Funcion principal

*****************************************************************************************************/

int main( int argc, char** argv )
{
 int i;
  /// Load an image
 for(i=0;i<MUESTRAS;i++){
	  src[i] = imread( argv[1] ); //Genero un vector de con copias de la foto original
	  Guardado[i]=src[i].clone(); //Genero un vector de con copias de respaldo

	  if( !src[i].data )
	  { return -1; }

	  imwrite("Resultados/original.jpg",Guardado[0]);
	
	  /// convierto imagen fuente en escala de grices
	  cvtColor( src[i], src_gray[i], COLOR_BGR2GRAY  );	 

	  //aplico filtros con "i" * iteraciones 
	  T_morfologica(src[i],i);
 
 	




	
  }

  Filtros(0 , 0); // aplico los filtros

 for(i=0;i<MUESTRAS;i++){
 	 Recorte_ancho(src_gray[i] , i); // recorto imagen por ancho

 	 cvtColor( src_recortada_1[i], src_recortada_1[i], COLOR_BGR2GRAY  );
	
	  Recorte_alto(src_recortada_1[i] , i); // recorto imagen  por alto
  }
  printf("\n\n\t\tFIN DEL PROGRAMA\n");
  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }

/*****************************************************************************************************

					Filtros

*****************************************************************************************************/


void Filtros(int, void*)
{
int i;
Mat AUX,element;
for(i=0;i<MUESTRAS;i++){
 

//***********************Filtro de Dilatacion**********************************//

 /* element = getStructuringElement( MORPH_RECT,  // Tipos: MORPH_RECT ;MORPH_CROSS;MORPH_ELLIPSE
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );

  dilate( src[i], src[i], element );
*/

//***********************Paso a escala de grices**********************************//
//
//cvtColor(fuente, destino,tipo cambio de color -> muchas macros como pa escribirlas)
//  CV_BGR2GRAY-> de RGB pasa a escala de grices 

cvtColor( src[i], src_gray[i], CV_BGR2GRAY   );

/// Reduce noise with a kernel 3x3

blur( src_gray[i], src_gray[i], Size(3,3) );  // (fuente , destino , tamaño)



//************************************** Filtro Canny	*****************************************
//
//Detector de bordes


//  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
// Canny( detected_edges, detected_edges, Minimo_Filtro, Maximo_Filtro*i, kernel_size );



//************************************** Detector de lineas*****************************************
//
// (Fuente, Destino,tipo,angulo y modulo,,,,)


//CvSeq* cvHoughLines2( src[i], AUX[i], CV_HOUGH_STANDARD, rho, theta, 10, 0,0 );




//***********************************   Umbralizacion   **************************************
//
// threshold(origen, destino, umbral,max_valor,tipo)
//Tipos:
//	THRESH_BINARY		-> Si el pixel es > al umbral setea este como Max_valor caso contraio 0
//	THRESH_BINARY_INV	-> Si el pixel es > al umbral setea este como 0 caso contraio Max_valor
//	THRESH_TRUNC		-> Si el pixel es > al umbral setea este como el umbralcaso contraio queda igual
//	THRESH_TOZERO		-> Cualquier pixel cuya intensidad no supere el umbral se establecerá a cero
//	THRESH_TOZERO_INV	-> Si la intensidad supera el umbral establecido el valor de salida se ajustará a cero
// 
//	Valor 0 es negro - Valor 255 es blanco
//	previo a aplicar la umbralizacion hay que pasar la foto a escala de grices
//********************************************************************************************

threshold( src_gray[i],AUX, 100+(10*i), 255,THRESH_BINARY); //(entrada, salida,umbral , maximo valor, tipo)



//threshold( AUX,AUX, 10, 255,THRESH_TOZERO_INV); //(entrada, salida,umbral , maximo valor, tipo)
//bitwise_not( detected_edges, dst );
//***********************************************************************************************

//Escritura de las 10 muestras

	if(i==0)
		imwrite("Resultados/Foto0.jpg",AUX);
	if(i==1)
		imwrite("Resultados/Foto1.jpg",AUX);	 
	if(i==2)
		imwrite("Resultados/Foto2.jpg",AUX);	 
	if(i==3)
		imwrite("Resultados/Foto3.jpg",AUX);	 
	if(i==4)
		imwrite("Resultados/Foto4.jpg",AUX);	 
	if(i==5)
		imwrite("Resultados/Foto5.jpg",AUX);	 
	if(i==6)
		imwrite("Resultados/Foto6.jpg",AUX);	 
	if(i==7)
		imwrite("Resultados/Foto7.jpg",AUX);	 
	if(i==8)
		imwrite("Resultados/Foto8.jpg",AUX);	 
	if(i==9)
		imwrite("Resultados/Foto9.jpg",AUX);

	


  }
 }

/*****************************************************************************************************

					Recorte de patente por ancho

*****************************************************************************************************/

void Recorte_ancho(Mat AUX ,int veces)
{
int i,contador=0;
size_t j,k,mayor=0;
Mat Resultado[MUESTRAS];
Mat blanco,negro,loquede;
Rect Rec_Mayor(0,0,0,0), Rec_Menor(0,0,0,0);
Rect Recorte(0,0,0,0);
Point Pto[4];

Mat Contornos_detectados = Mat::zeros( src[1].size(), CV_8UC3 );
blanco=Mat::ones(src[1].size(),CV_8UC3);
negro=Mat::zeros(src[1].size(),CV_8UC3);
loquede=src[1].colRange(0,(src[1].cols)/2);

//absdiff(src[1],blanco,src_gray[1]);

//threshold( src_gray[1],src_gray[1], 200, 255,THRESH_BINARY); //(entrada, salida,umbral , maximo valor, tipo)	

//for(i=0;i<MUESTRAS;i++)
//{


	blur( AUX,AUX, Size(3,3) );  

	threshold( AUX,AUX, 120, 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)



//	threshold( src_gray[i],src_gray[i], 100+(10*i), 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)

//	AUX=src_gray[1].clone();//trabajo con AUX y guardo src_gray como muestra original
 

//Ver si reconoce al rectangulo sin el filtro canny
	//Canny(AUX[i], AUX[i], 100, 200 * 2); 
	
	//Canny(src_gray[i], src_gray[i], 100, 200 * 2); 
	//threshold( src_gray[i],src_gray[i], 100+(10*i), 255,	THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)


	vector<vector<Point> > contours;
	CvSeq* secuencia_ptos;
	vector<Vec4i> hierarchy;
	CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours
	 


//*******/
//CV_RETR_EXTERNAL,CV_RETR_LIST,CV_RETR_CCOMP,CV_RETR_TREE
//



	findContours(AUX, contours, hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        i=0;
	j=0;

  for(  k = 0; k < contours.size(); k++ )
	if(contours[k].size())	
	{  
	contador=contours[k].size();		
	printf(" \n\n\t\t %d° Contador= %d",i,contador);
	i++;
	//filtro por mas grande
//	if(contours[k].size())	
		Recorte=boundingRect(contours[k]); //saco rectangulo de un contorn0


	if(Recorte.width>Rec_Mayor.width) //comparo el ancho del rectangulo obtenido con el mayor guardado
	   if(contours[k].size()>500)
	   if(contours[k].size()<1000)
		{	
		mayor=k;			//guardo que posicion del ancho nuevo
		Rec_Mayor.width=Recorte.width;	//guardo el ancho mayor nuevo
		Rec_Mayor=Recorte;		//guardo el alto menor
		}
	}

	printf("\n\n\t el mayor es el %d y el ancho es %d",mayor,Rec_Mayor.width);


       //imprimo contornos filtrados
   for(  k = 0; k < contours.size(); k++ )
	if(contours[k].size())
	{
	if(contours[k].size()>000)
	   if(contours[k].size()<2000){
	     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	     drawContours( Contornos_detectados, contours, k, color, 2, 8, hierarchy, 0, Point() );
		}        
	}

//	src_recortada_1[veces]=Guardado[veces](Rec_Mayor);
	src_recortada_1[veces]=src[veces](Rec_Mayor);
//cvtColor(  src_recortada_1[veces],src_recortada_1[veces], CV_BGR2GRAY   );
//
threshold( src_recortada_1[veces],src_recortada_1[veces], 120, 255,THRESH_BINARY_INV);

//Imprimo las imagenes recortadas
if(veces==0){
	imwrite( "Resultados/Recorte_0_1.jpg",src_recortada_1[veces]);
        imwrite("Resultados/Contornos_detectados_0_1.jpg",Contornos_detectados);}
if(veces==1){
	imwrite( "Resultados/Recorte_1_1.jpg",src_recortada_1[veces]);
        imwrite( "Resultados/Contornos_detectados_1_1.jpg",Contornos_detectados);}        
if(veces==2){
	imwrite( "Resultados/Recorte_2_1.jpg",src_recortada_1[veces]);
        imwrite("Resultados/Contornos_detectados_2_1.jpg",Contornos_detectados);}
if(veces==3){
	imwrite( "Resultados/Recorte_3_1.jpg",src_recortada_1[veces]);
        imwrite( "Resultados/Contornos_detectados_3_1.jpg",Contornos_detectados);}	
if(veces==4){
	imwrite("Resultados/Recorte_4_1.jpg",src_recortada_1[veces]);
        imwrite("Resultados/Contornos_detectados_4_1.jpg",Contornos_detectados);}
if(veces==5){
	imwrite("Resultados/Recorte_5_1.jpg",src_recortada_1[veces]);
        imwrite( "Resultados/Contornos_detectados_5_1.jpg",Contornos_detectados);}
if(veces==6){
	imwrite( "Resultados/Recorte_6_1.jpg",src_recortada_1[veces]);
        imwrite( "Resultados/Contornos_detectados_6_1.jpg",Contornos_detectados);}        
if(veces==7){
	imwrite("Resultados/Recorte_7_1.jpg",src_recortada_1[veces]);
        imwrite( "Resultados/Contornos_detectados_7_1.jpg",Contornos_detectados);}
if(veces==8){
	imwrite("Resultados/Recorte_8_1.jpg",src_recortada_1[veces]);
        imwrite("Resultados/Contornos_detectados_8_1.jpg",Contornos_detectados);}	
if(veces==9){
	imwrite( "Resultados/Recorte_9_1.jpg",src_recortada_1[veces]);
        imwrite( "Resultados/Contornos_detectados_9_1.jpg",Contornos_detectados);}

//src[veces]=src_recortada.clone();

		

//} // fin del for de MUESTRAS


}

/*****************************************************************************************************

					Recorte de patente por altura

*****************************************************************************************************/

void Recorte_alto(Mat AUX ,int veces)
{
int i,contador=0;
size_t j,k,mayor=0;
Mat Resultado[MUESTRAS];
Mat blanco,negro,loquede;
Rect Rec_Mayor(0,0,0,0),Recorte(0,0,0,0);
Point Pto[4];

Mat Contornos_detectados = Mat::zeros( src[1].size(), CV_8UC3 );
blanco=Mat::ones(src[1].size(),CV_8UC3);
negro=Mat::zeros(src[1].size(),CV_8UC3);
loquede=src[1].colRange(0,(src[1].cols)/2);

//absdiff(src[1],blanco,src_gray[1]);

//threshold( src_gray[1],src_gray[1], 200, 255,THRESH_BINARY); //(entrada, salida,umbral , maximo valor, tipo)	

//for(i=0;i<MUESTRAS;i++)
//{


	blur( AUX,AUX, Size(3,3) );  

	threshold( AUX,AUX, 120, 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)



//	threshold( src_gray[i],src_gray[i], 100+(10*i), 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)

//	AUX=src_recortada.clone();//trabajo con AUX y guardo src_gray como muestra original
 

//Ver si reconoce al rectangulo sin el filtro canny
	//Canny(AUX[i], AUX[i], 100, 200 * 2); 
	
	//Canny(src_gray[i], src_gray[i], 100, 200 * 2); 
	//threshold( src_gray[i],src_gray[i], 100+(10*i), 255,	THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)


	vector<vector<Point> > contours;
	CvSeq* secuencia_ptos;
	vector<Vec4i> hierarchy;
	CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours
	 


//*******/
//CV_RETR_EXTERNAL,CV_RETR_LIST,CV_RETR_CCOMP,CV_RETR_TREE
//



	findContours(AUX, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        i=0;
	j=0;

   for(  k = 0; k < contours.size(); k++ )
	if(contours[k].size())	
	{  
	contador=contours[k].size();		
	printf(" \n\n\t\t %d° Contador= %d",i,contador);
	i++;
	//filtro por mas grande
	//if(contours[k].size())
		Recorte=boundingRect(contours[k]); //saco rectangulo de un contorn0
	if(Recorte.height>Rec_Mayor.height) //comparo el ancho del rectangulo obtenido con el mayor guardado

		{	
		mayor=k;			//guardo que posicion del ancho nuevo
		Rec_Mayor.height=Recorte.height;	//guardo el ancho mayor nuevo
		Rec_Mayor=Recorte;
		}
	}

	printf("\n\n\t el mayor es el %d y el alto es %d\n",mayor,Rec_Mayor.height);
       //imprimo contornos filtrados
   for(  k = 0; k < contours.size(); k++ )
	if(contours[k].size())
	{

	if(contours[k].size()>0)
	   if(contours[k].size()<1000){
	     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	     drawContours( Contornos_detectados, contours, k, color, 2, 8, hierarchy, 0, Point() );
		}
        
	}
src_recortada_2[veces]=src[veces](Rec_Mayor);

//Imprimo las imagenes recortadas
if(veces==0){
	imwrite( "Resultados/Recorte_0_2.jpg",src_recortada_2[veces]);
        imwrite("Contornos_detectados_0_2.jpg",Contornos_detectados);}
if(veces==1){
	imwrite( "Resultados/Recorte_1_2.jpg",src_recortada_2[veces]);
        imwrite( "Contornos_detectados_1_2.jpg",Contornos_detectados);}        
if(veces==2){
	imwrite( "Resultados/Recorte_2_2.jpg",src_recortada_2[veces]);
        imwrite("Contornos_detectados_2_2.jpg",Contornos_detectados);}
if(veces==3){
	imwrite( "Resultados/Recorte_3_2.jpg",src_recortada_2[veces]);
        imwrite( "Contornos_detectados_3_2.jpg",Contornos_detectados);}	
if(veces==4){
	imwrite("Resultados/Recorte_4_2.jpg",src_recortada_2[veces]);
        imwrite("Contornos_detectados_4_2.jpg",Contornos_detectados);}
if(veces==5){
	imwrite("Resultados/Recorte_5_2.jpg",src_recortada_2[veces]);
        imwrite( "Contornos_detectados_5_2.jpg",Contornos_detectados);}
if(veces==6){
	imwrite( "Resultados/Recorte_6_2.jpg",src_recortada_2[veces]);
        imwrite( "Contornos_detectados_6_2.jpg",Contornos_detectados);}        
if(veces==7){
	imwrite("Resultados/Recorte_7_2.jpg",src_recortada_2[veces]);
        imwrite( "Contornos_detectados_7_2.jpg",Contornos_detectados);}
if(veces==8){
	imwrite("Resultados/Recorte_8_2.jpg",src_recortada_2[veces]);
        imwrite("Contornos_detectados_8_2.jpg",Contornos_detectados);}	
if(veces==9){
	imwrite( "Resultados/Recorte_9_2.jpg",src_recortada_2[veces]);
        imwrite( "Contornos_detectados_9_2.jpg",Contornos_detectados);}


//} // fin del for de MUESTRAS


}


/*****************************************************************************************************

					trasformaciones morfologicas

*****************************************************************************************************/


void T_morfologica(Mat AUX ,int veces)
{
	int iteraciones=5;
	Mat kernel;
	kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

	threshold( AUX,AUX, 120, 255,THRESH_BINARY_INV);

//	erode(AUX,AUX,kernel,Point(-1,-1),iteraciones*veces,BORDER_CONSTANT,morphologyDefaultBorderValue());
//	dilate(AUX,AUX,kernel,Point(-1,-1),iteraciones*veces,BORDER_CONSTANT,morphologyDefaultBorderValue());

	morphologyEx(AUX,AUX,MORPH_OPEN,kernel,Point(-1,-1),iteraciones*veces,BORDER_CONSTANT,morphologyDefaultBorderValue());
	morphologyEx(AUX,AUX,MORPH_CLOSE,kernel,Point(-1,-1),iteraciones*veces,BORDER_CONSTANT,morphologyDefaultBorderValue());



}



/*****************************************************************************************************

					Lectura de patentes

*****************************************************************************************************/


void Lectura(int ,void*)
{

vector<Mat> canales;
//computeNMChannels(src_recortada, canales);

//run(src_recortada,canales);
}
