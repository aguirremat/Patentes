//#include "/home/matias/proyecto/opencv-2.4.13.4/release/opencv_contrib/modules/text/include/opencv2/text/erfilter.hpp"
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



#define MUESTRAS 10

using namespace cv;
using namespace cv::text;

//	funciones   	 //

void Filtros(int, void*);
void Recorte_ancho(int, void*);
void Recorte_alto(int, void*);
void Lectura(int ,void*);
void T_morfologica(int ,void*);

//	Global variables //

Mat src[MUESTRAS],src_gray[MUESTRAS];
Mat dst, detected_edges;



int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Patente";
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
Mat src_recortada;
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
	
  if( !src[i].data )
  { return -1; }
  }

 
 

//cvtColor(src[0], src[0], COLOR_BGR2Luv);

for(i=0;i<MUESTRAS;i++){
  /// Create a matrix of the same type and size as src (for dst)
  dst.create(src[i].size(), src[i].type() );
 
  /// Convert the image to grayscale
  cvtColor( src[i], src_gray[i], COLOR_BGR2GRAY  );
  }
	
  Filtros(0 , 0); // aplico los filtros
  Recorte_ancho(0 , 0); // recorto imagen  

  cvtColor( src_recortada, src_recortada, COLOR_BGR2GRAY  );
	
  T_morfologica (0,0);

//  Filtros(0 , 0); // aplico los filtros
  Recorte_alto(0 , 0); // recorto imagen  

  
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


  /// Using Canny's output as a mask, we display our result
 // dst = Scalar::all(0);

 //src[i].copyTo( dst, AUX);

//Escritura de los 10 ciclos
if(i==0)
  imwrite( "Foto0.jpg" , src_gray[0]);
if(i==1)
  imwrite( "Foto1.jpg", AUX);
if(i==2)
  imwrite( "Foto2.jpg", AUX);
if(i==3)
  imwrite( "Foto3.jpg", AUX);
if(i==4)
  imwrite( "Foto4.jpg", AUX);
if(i==5)
  imwrite( "Foto5.jpg", AUX);	
if(i==6)
  imwrite( "Foto6.jpg", AUX);
if(i==7)
  imwrite( "Foto7.jpg", AUX);
if(i==8)
  imwrite( "Foto8.jpg", AUX);
if(i==9)
  imwrite( "Foto9.jpg", AUX);
 // imshow window_name, dst );
  }
 }



/*****************************************************************************************************

					Recorte de patente por ancho

*****************************************************************************************************/

void Recorte_ancho(int ,void*)
{
int i,contador=0;
size_t j,k,mayor=0;
Mat AUX;
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


	blur( src_gray[1], src_gray[1], Size(3,3) );  

	threshold( src_gray[1],src_gray[1], 120, 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)



//	threshold( src_gray[i],src_gray[i], 100+(10*i), 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)

	AUX=src_gray[1].clone();//trabajo con AUX y guardo src_gray como muestra original
 

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
	{  
	contador=contours[k].size();		
	printf(" \n\n\t\t %d° Contador= %d",i,contador);
	i++;
	//filtro por mas grande
	Recorte=boundingRect(contours[k]); //saco rectangulo de un contorn0
	if(Recorte.width>Rec_Mayor.width) //comparo el ancho del rectangulo obtenido con el mayor guardado
		{	
		mayor=k;			//guardo que posicion del ancho nuevo
		Rec_Mayor.width=Recorte.width;	//guardo el ancho mayor nuevo
		Rec_Mayor=Recorte;
		}
	}

	printf("\n\n\t el mayor es el %d",mayor);
       //imprimo contornos filtrados
   for(  k = 0; k < contours.size(); k++ )
	{

	if(contours[k].size()>0)
	   if(contours[k].size()<1000){
	     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	     drawContours( Contornos_detectados, contours, k, color, 2, 8, hierarchy, 0, Point() );
		}
        
	}
	src_recortada=src[1](Rec_Mayor);
	imwrite( "Recorte_1.jpg",src_recortada);
        imwrite( "Remarco.jpg", AUX);
        imwrite( "Contornos_detectados.jpg", Contornos_detectados);
	imwrite( "Control.jpg", src_gray[1]);

		

//} // fin del for de MUESTRAS


}

/*****************************************************************************************************

					Recorte de patente por altura

*****************************************************************************************************/

void Recorte_alto(int ,void*)
{
int i,contador=0;
size_t j,k,mayor=0;
Mat AUX;
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


	blur( src_recortada,src_recortada, Size(3,3) );  

	threshold( src_recortada,src_recortada, 120, 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)





//	threshold( src_gray[i],src_gray[i], 100+(10*i), 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)

	AUX=src_recortada.clone();//trabajo con AUX y guardo src_gray como muestra original
 

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
	{  
	contador=contours[k].size();		
	printf(" \n\n\t\t %d° Contador= %d",i,contador);
	i++;
	//filtro por mas grande
	Recorte=boundingRect(contours[k]); //saco rectangulo de un contorn0
	if(Recorte.height>Rec_Mayor.height) //comparo el ancho del rectangulo obtenido con el mayor guardado
		{	
		mayor=k;			//guardo que posicion del ancho nuevo
		Rec_Mayor.height=Recorte.height;	//guardo el ancho mayor nuevo
		Rec_Mayor=Recorte;
		}
	}

	printf("\n\n\t el mayor es el %d\n",mayor);
       //imprimo contornos filtrados
   for(  k = 0; k < contours.size(); k++ )
	{

	if(contours[k].size()>0)
	   if(contours[k].size()<1000){
	     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	     drawContours( Contornos_detectados, contours, k, color, 2, 8, hierarchy, 0, Point() );
		}
        
	}
	src_recortada=src_recortada(Rec_Mayor);
	imwrite( "Recorte_2.jpg",src_recortada);
        imwrite( "Remarco.jpg", AUX);
        imwrite( "Contornos_detectados.jpg", Contornos_detectados);
	imwrite( "Control.jpg", src_gray[1]);

		

//} // fin del for de MUESTRAS


}


/*****************************************************************************************************

					trasformaciones morfologicas

*****************************************************************************************************/


void T_morfologica(int ,void*)
{
	int iteraciones=10;

	threshold( src_recortada,src_recortada, 120, 255,THRESH_BINARY_INV);

	//erode(src_recortada,src_recortada,(3,3),Point(-1,-1),iteraciones,BORDER_CONSTANT,morphologyDefaultBorderValue());

//	Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

//	morphologyEx(src_recortada,src_recortada,MORPH_OPEN,(3,3),Point(0,0),1,BORDER_CONSTANT,morphologyDefaultBorderValue());
//	morphologyEx(src_recortada,src_recortada,MORPH_CLOSE,(3,3),Point(0,0),1,BORDER_CONSTANT,morphologyDefaultBorderValue());
//	morphologyEx(src_gray[i],src_gray[i],MORPH_CLOSE,element );

	imwrite( "Recorte_3.jpg",src_recortada);
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












