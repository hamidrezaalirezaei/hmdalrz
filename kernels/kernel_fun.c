/* --------------------------------------------------------------------
 Evaluation of kernel functions: linear, poly, rbf, sigmoidal, Wavelet.

 -------------------------------------------------------------------- */

#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <string.h>

/* --- Global variables --------------------------------------------- */

double *dataA;      /* pointer at the fist patterns */
double *dataB;      /* pointer at the second patterns */
long dim;           /* dimension of patterns */
int ker;            /* kernel id (0 - linear, 1 - polynomial, 
                       2 - rbf, 3 - sigmoid, 4 - Wavelet */
double *arg1;       /* kernel argument */
long ker_cnt;       /* number of cernel evaluations */

/* "custom" can be replaced by user's own kernel-identifier */
char *kernel_name[] = {"linear","poly","rbf","sigmoid","wavelet"};

/* -------------------------------------------------------------------
 Computes dot product of subtraction of a-th and b-th vector.
 c = (a-b)'*(a-b)
------------------------------------------------------------------- */
double sub_dot_prod( long a, long b )
{
   double c = 0;
   long i;
   for( i = 0; i < dim; i++ ) {
      c += (*(dataA+(a*dim)+i) - *(dataB+(b*dim)+i))*
           (*(dataA+(a*dim)+i) - *(dataB+(b*dim)+i));
   }
   return( c );
}


/* -------------------------------------------------------------------
 Computes dot product of a-th and b-th vector.
 c = a'*b
------------------------------------------------------------------- */
double dot_prod( long a, long b)
{
   double c = 0;
   long i;
   for( i = 0; i < dim; i++ ) {
      c += *(dataA+(a*dim)+i) * *(dataB+(b*dim)+i);
   }
   return( c );
}

double myfunc( long a, long b)
{
     long i;
     double nrm;
     double aa;	
     double c=1;
     for( i = 0; i < dim; i++ ) {
         nrm =     ( (*(dataA+(a*dim)+i) - *(dataB+(b*dim)+i)) *
                         (*(dataA+(a*dim)+i) - *(dataB+(b*dim)+i)) );
         
         aa = cos( ( 1.75 * (*(dataA+(a*dim)+i) - *(dataB+(b*dim)+i)) ) /arg1[0] ) * 
                exp( -0.5 * nrm /(arg1[0]*arg1[0]) ) ;
         
         c = c * aa;
     }
     return( c );
}

/* --------------------------------------------------------------------
 Converts string kernel identifier to int.
-------------------------------------------------------------------- */
int kernel_id( const mxArray *prhs1 )
{
  int num, i, buf_len;
  char *buf;

  if( mxIsChar( prhs1 ) != 1) return( -1 );

  buf_len  = (mxGetM(prhs1) * mxGetN(prhs1)) + 1;
  buf = mxCalloc( buf_len, sizeof( char ));

  mxGetString( prhs1, buf, buf_len );
  
  num = sizeof( kernel_name )/sizeof( char * );

  for( i = 0; i < num; i++ ) {
    if( strcmp( buf, kernel_name[i] )==0 ) return( i );
  }

  return(-1);
}

/* --------------------------------------------------------------------
 Computes kernel function for a-th and b-th.

 The base address for the 1st argument is dataA and for the 2nd
 argument is dataB.
-------------------------------------------------------------------- */
double kernel( long a, long b )
{
   double c = 0;

   ker_cnt++;

   switch( ker ) {
      /* linear kernel */
      case 0:
         c = dot_prod( a, b );
         break;
      /* polynomial kernel */
      case 1:
         c = pow( (dot_prod( a, b) + arg1[0]), arg1[1] );
//            c = pow( (arg1[0]*dot_prod( a, b) + 3), arg1[1] );
         break;
      /* radial basis functions kernel*/
      case 2:
         c = exp( -0.5*sub_dot_prod( a, b)/(arg1[0]*arg1[0]) );
         break;
      /* sigmoid */
      case 3: 
		c = tanh( arg1[0]*dot_prod( a,b) + arg1[1] );
         break;
      /* "wavelet"*/
      case 4:     
         c = myfunc (a,b);
         break;
      default:
         c = 0;
   }
   return( c );
}

