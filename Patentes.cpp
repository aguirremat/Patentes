#include "/home/matias/proyecto/opencv-2.4.13.4/release/opencv_contrib/modules/text/include/opencv2/text.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>


#define MUESTRAS 10

using namespace cv;

//	funciones   	 //

void Filtros(int, void*);
void Recorte(int, void*);

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
	

  //Recorte(0 , 0); // recorto imagen  
  Filtros(0 , 0); // aplico los filtros
  
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

  element = getStructuringElement( MORPH_RECT,  // Tipos: MORPH_RECT ;MORPH_CROSS;MORPH_ELLIPSE
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );

  dilate( src[i], src[i], element );


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

threshold( src_gray[i],AUX, 100+(10*i), 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)



//threshold( AUX,AUX, 10, 255,THRESH_TOZERO_INV); //(entrada, salida,umbral , maximo valor, tipo)
//bitwise_not( detected_edges, dst );
//***********************************************************************************************


  /// Using Canny's output as a mask, we display our result
 // dst = Scalar::all(0);

 //src[i].copyTo( dst, AUX);

//Escritura de los 10 ciclos
if(i==0)
  imwrite( "Foto0.jpg", src_gray[0]);
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

					Recorte de patente

*****************************************************************************************************/

void Recorte(int ,void*)
{
int i;
size_t j,k,CUATRO;
Mat AUX[MUESTRAS];
Mat Resultado[MUESTRAS];
Rect Recorte;
Point Pto[20];

for(i=0;i<MUESTRAS;i++)
{
	cvtColor( src[i], AUX[i], CV_BGR2GRAY   );

	blur( AUX[i], AUX[i], Size(3,3) );  

	threshold( AUX[i],AUX[i], 100+(10*i), 255,THRESH_BINARY_INV); //(entrada, salida,umbral , maximo valor, tipo)

 
//Ver si reconoce al rectangulo sin el filtro canny
//Canny(src_gray[i], canny_output, 100, 200 * 2); 
	
	vector<vector<Point> > contours;
	CvSeq* secuencia_ptos;
	vector<Vec4i> hierarchy;
	CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours
	 

	findContours(AUX[i], contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        
   for(  k = 0; k < contours.size(); k++ )
	{
	approxPolyDP(contours[k],contours[k],3,true);
	//filtramos los contornos distintos de 4
	if(contours[k].size()==4) 
	for(j=0;j<contours[k].size();j++)
		{
		//De aca saco los puntos para recortar


		}	

		
	}
		

      
   	
	rectangle(AUX[i], Recorte, cv::Scalar(0, 255, 0));


        imwrite( "Recorte.jpg", AUX[5]);
	//if(Recorte.width > 0 && Recorte.height>0)
	//	cv::imshow("imagen_recortada",src[i](Recorte));


} // fin del for de MUESTRAS


}
















