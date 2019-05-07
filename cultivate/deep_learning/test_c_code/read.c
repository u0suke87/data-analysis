#include <stdio.h>
#include <stdlib.h>

int main(void){
	
	FILE *fp;
	int ret,a,b;
	float i,j,k;

	fp = fopen( "data.csv", "r");
	while( (ret = fscanf(fp, "%d,%f,%f,%f,%d\n", &a, &i, &j, &k, &b)) != EOF){
		printf("%d,%f,%f,%f,%d\n", a, i, j, k, b);
	}

	return 0;
}

