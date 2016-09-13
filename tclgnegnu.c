/* Tcl bindings for gnegnu neural networks.
 * Copyright (C) 2003 Salvatore Sanfilippo <antirez@invece.org>
 * All rights reserved.
 *
 * See LICENSE for Copyright and License information. */

#include <tcl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

#define VERSION "0.1"

/* -------------------------- ANN object implementation --------------------- */

static void Tcl_SetAnnObj(Tcl_Obj *objPtr, struct Ann *srcnet);
static void FreeAnnInternalRep(Tcl_Obj *objPtr);
static void DupAnnInternalRep(Tcl_Obj *srcPtr, Tcl_Obj *copyPtr);
static void UpdateStringOfAnn(Tcl_Obj *objPtr);
static int SetAnnFromAny(struct Tcl_Interp* interp, Tcl_Obj *objPtr);

struct Tcl_ObjType tclAnnType = {
	"ann",
	FreeAnnInternalRep,
	DupAnnInternalRep,
	UpdateStringOfAnn,
	SetAnnFromAny
};

/* This function set objPtr as an ann object with value
 * 'val'. If 'val' == NULL, the object is set to an empty net. */
void Tcl_SetAnnObj(Tcl_Obj *objPtr, struct Ann *srcnet)
{
	Tcl_ObjType *typePtr;
	struct Ann *net;

	/* It's not a good idea to set a shared object... */
	if (Tcl_IsShared(objPtr)) {
		panic("Tcl_SetMpzObj called with shared object");
	}
	/* Free the old object private data and invalidate the string
	 * representation. */
	typePtr = objPtr->typePtr;
	if ((typePtr != NULL) && (typePtr->freeIntRepProc != NULL)) {
		(*typePtr->freeIntRepProc)(objPtr);
	}
	Tcl_InvalidateStringRep(objPtr);
	/* Allocate and initialize a new neural network */
	if (srcnet) {
		net = AnnClone(srcnet);
	} else {
		net = AnnAlloc(0);
	}
	if (!net) {
		panic("Out of memory in Tcl_SetAnnObj");
	}
	/* Set it as object private data, and type */
	objPtr->typePtr = &tclAnnType;
	objPtr->internalRep.otherValuePtr = (void*) net;
}

/* Return an ANN from the object. If the object is not of type ANN
 * an attempt to convert it to ANN is done. On failure (the string
 * representation of the object can't be converted to a neural net)
 * an error is returned. */
int Tcl_GetAnnFromObj(struct Tcl_Interp *interp, Tcl_Obj *objPtr, struct Ann **annpp)
{
	int result;

	if (objPtr->typePtr != &tclAnnType) {
		result = SetAnnFromAny(interp, objPtr);
		if (result != TCL_OK)
			return result;
	}
	*annpp = (struct Ann*) objPtr->internalRep.longValue;
	return TCL_OK;
}

/* The 'free' method of the object. */
void FreeAnnInternalRep(Tcl_Obj *objPtr)
{
	struct Ann* net= (struct Ann*) objPtr->internalRep.otherValuePtr;

	AnnFree(net);
}

/* The 'dup' method of the object */
void DupAnnInternalRep(Tcl_Obj *srcPtr, Tcl_Obj *copyPtr)
{
	struct Ann *annCopyPtr, *annSrcPtr;

	annSrcPtr = (struct Ann*) srcPtr->internalRep.otherValuePtr;
	if ((annCopyPtr = AnnClone(annSrcPtr)) == NULL)
		panic("Out of memory inside DupMpzInternalRep()");
	copyPtr->internalRep.otherValuePtr = (void*) annCopyPtr;
	copyPtr->typePtr = &tclAnnType;
}

/* Helper function for UpdateStringOfAnn() function */
static void StrAppendListDouble(char **pptr, double *v, int len)
{
	char *b = *pptr;
	int i;

	*b++ = '{';
	if (v) {
		for (i = 0; i < len; i++) {
			char aux[64];
			int l;

			sprintf(aux, "%.16f ", v[i]);
			l = strlen(aux);
			memcpy(b, aux, l);
			b += l;
			*b++ = ' ';
		}
	}
	*b++ = '}';
	*b++ = ' ';
	*pptr = b;
}

/* The 'update string' method of the object */
void UpdateStringOfAnn(Tcl_Obj *objPtr)
{
	struct Ann *net = (struct Ann*) objPtr->internalRep.otherValuePtr;
	double aux[6];
	size_t len = 0;
	char *b, *algostr;
	int j;

	/* Guess how many bytes are needed for the representation */
	for (j = 0; j < LAYERS(net); j++) {
		int units = UNITS(net,j);
		int weights;
		weights = j == 0 ? 0 : WEIGHTS(net,j);
		/* output and error array */
		len += 2 * 24 * units;
		/* weight, gradient, pgradient, delta, sgradient */
		len += 5 * 24 * weights;
		/* list delimiters and spaces */
		len += 8 * 3;
	}
	len += (6 * 24) + 5; /* Final list of parameters */
	len += 64; /* flags */
	objPtr->bytes = ckalloc(len);
	b = (char *) objPtr->bytes;
	/* Convert to string */
	for (j = 0; j < LAYERS(net); j++) {
		int units = UNITS(net,j);
		int weights;
		weights = j == 0 ? 0 : WEIGHTS(net,j);
		*b++ = '{';
		StrAppendListDouble(&b, net->layer[j].output, units);
		StrAppendListDouble(&b, net->layer[j].error, units);
		StrAppendListDouble(&b, net->layer[j].weight, weights);
		StrAppendListDouble(&b, net->layer[j].gradient, weights);
		StrAppendListDouble(&b, net->layer[j].pgradient, weights);
		StrAppendListDouble(&b, net->layer[j].delta, weights);
		StrAppendListDouble(&b, net->layer[j].sgradient, weights);
		*b++ = '}';
		*b++ = ' ';
	}
	/* Net configuration */
	*b++ = '{';
	aux[0] = net->learn_rate;
	aux[1] = net->momentum;
	aux[2] = net->rprop_nminus;
	aux[3] = net->rprop_nplus;
	aux[4] = net->rprop_maxupdate;
	aux[5] = net->rprop_minupdate;
	StrAppendListDouble(&b, aux, 6);
	*b++ = '}';
	*b++ = ' ';
	/* Net flags */
	switch(net->flags & ANN_ALGOMASK) {
	case ANN_BBPROP:	algostr = "bbprop"; break;
	case ANN_OBPROP:	algostr = "obprop"; break;
	case ANN_BBPROPM:	algostr = "bbpropm"; break;
	case ANN_OBPROPM:	algostr = "obpropm"; break;
	case ANN_RPROP:		algostr = "rprop"; break;
	default: algostr = "unknown"; break;
	}
	memcpy(b, algostr, strlen(algostr));
	b += strlen(algostr);
	*b = '\0';
	objPtr->length = strlen(objPtr->bytes);
}

/* The 'set from any' method of the object */
int SetAnnFromAny(struct Tcl_Interp* interp, Tcl_Obj *objPtr)
{
#if 0
	char *s;
	mpz_t t;
	mpz_ptr mpzPtr;
	Tcl_ObjType *typePtr;

	if (objPtr->typePtr == &tclMpzType)
		return TCL_OK;

	/* Try to convert */
	s = Tcl_GetStringFromObj(objPtr, NULL);
	mpz_init(t);
	if (mpz_set_str(t, s, 0) != SBN_OK) {
		mpz_clear(t);
		Tcl_ResetResult(interp);
		Tcl_AppendStringsToObj(Tcl_GetObjResult(interp),
				"Invalid big number: \"",
				s, "\" must be a relative integer number",
				NULL);
		return TCL_ERROR;
	}
	/* Allocate */
	mpzPtr = (mpz_ptr) ckalloc(sizeof(struct struct_sbnz));
	mpz_init(mpzPtr);
	/* Free the old object private rep */
	typePtr = objPtr->typePtr;
	if ((typePtr != NULL) && (typePtr->freeIntRepProc != NULL)) {
		(*typePtr->freeIntRepProc)(objPtr);
	}
	/* Set it */
	objPtr->typePtr = &tclMpzType;
	objPtr->internalRep.otherValuePtr = (void*) mpzPtr;
	memcpy(mpzPtr, t, sizeof(*mpzPtr));
	return TCL_OK;
#endif
	Tcl_AppendStringsToObj(Tcl_GetObjResult(interp),
			"SetAnnFromAny() not implemented", NULL);
	return TCL_ERROR;
}

/* --------------- the actual commands for multipreicision math ------------- */

#if 0
static int BigBasicObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	Tcl_Obj *result;
	mpz_t res;
	mpz_ptr t;
	char *cmd;

	cmd = Tcl_GetStringFromObj(objv[0], NULL);
	objc--;
	objv++;

	result = Tcl_GetObjResult(interp);
	mpz_init(res);
	mpz_setzero(res);
	if (cmd[0] == '*' || cmd[0] == '/') {
		if (mpz_set_ui(res, 1) != SBN_OK)
			goto err;
	}
	if ((cmd[0] == '/' || cmd[0] == '%' || cmd[0] == '-') && objc) {
		if (Tcl_GetMpzFromObj(interp, objv[0], &t) != TCL_OK)
			goto err;
		if (mpz_set(res, t) != SBN_OK)
			goto oom;
		if (cmd[0] == '-' && objc == 1)
			res->s = !res->s;
		objc--;
		objv++;
	}
	while(objc--) {
		if (Tcl_GetMpzFromObj(interp, objv[0], &t) != TCL_OK)
			goto err;
		switch(cmd[0]) {
		case '+':
			if (mpz_add(res, res, t) != SBN_OK)
				goto oom;
			break;
		case '-':
			if (mpz_sub(res, res, t) != SBN_OK)
				goto oom;
			break;
		case '*':
			if (mpz_mul(res, res, t) != SBN_OK)
				goto oom;
			break;
		case '/':
			if (mpz_tdiv_q(res, res, t) != SBN_OK)
				goto oom;
			break;
		case '%':
			if (mpz_mod(res, res, t) != SBN_OK)
				goto oom;
			break;
		}
		objv++;
	}
	Tcl_SetMpzObj(result, res);
	mpz_clear(res);
	return TCL_OK;
err:
	mpz_clear(res);
	return TCL_ERROR;
oom:
	Tcl_SetStringObj(result, "Out of memory doing multiprecision math", -1);
	mpz_clear(res);
	return TCL_ERROR;
}

static int BigCmpObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	Tcl_Obj *result;
	mpz_ptr a, b;
	int cmp, res;
	char *cmd;

	if (objc != 3) {
		Tcl_WrongNumArgs(interp, 1, objv, "bignum bignum");
		return TCL_ERROR;
	}

	cmd = Tcl_GetStringFromObj(objv[0], NULL);
	if (Tcl_GetMpzFromObj(interp, objv[1], &a) != TCL_OK ||
	    Tcl_GetMpzFromObj(interp, objv[2], &b) != TCL_OK)
		return TCL_ERROR;
	cmp = mpz_cmp(a, b);

	result = Tcl_GetObjResult(interp);
	res = 0;
	switch(cmd[0]) {
	case '>':
		switch(cmd[1]) {
		case '=':
			if (cmp >= 0) res = 1;
			break;
		default:
			if (cmp > 0) res = 1;
			break;
		}
		break;
	case '<':
		switch(cmd[1]) {
		case '=':
			if (cmp <= 0) res = 1;
			break;
		default:
			if (cmp < 0) res = 1;
			break;
		}
		break;
	case '=':
		if (cmp == 0) res = 1;
		break;
	case '!':
		if (cmp != 0) res = 1;
		break;
	}
	Tcl_SetIntObj(result, res);
	return TCL_OK;
}

static int BigRandObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	Tcl_Obj *result;
	int len = 1;
	mpz_t r;

	if (objc != 1 && objc != 2) {
		Tcl_WrongNumArgs(interp, 1, objv, "?atoms?");
		return TCL_ERROR;
	}
	if (objc == 2 && Tcl_GetIntFromObj(interp, objv[1], &len) != TCL_OK)
		return TCL_ERROR;
	result = Tcl_GetObjResult(interp);
	mpz_init(r);
	if (mpz_random(r, len) != SBN_OK) {
		mpz_clear(r);
		Tcl_SetStringObj(result, "Out of memory", -1);
		return TCL_ERROR;
	}
	Tcl_SetMpzObj(result, r);
	mpz_clear(r);
	return TCL_OK;
}

static int BigSrandObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	char *seed;
	int len;

	if (objc != 2) {
		Tcl_WrongNumArgs(interp, 1, objv, "seed-string");
		return TCL_ERROR;
	}
	seed = Tcl_GetStringFromObj(objv[1], &len);
	sbn_seed(seed, len);
	return TCL_OK;
}

static int BigPowObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	Tcl_Obj *result;
	int mpzerr;
	mpz_t r; /* result */
	mpz_ptr b, e, m; /* base, exponent, modulo */

	if (objc != 3 && objc != 4) {
		Tcl_WrongNumArgs(interp, 1, objv, "base exponent ?modulo?");
		return TCL_ERROR;
	}
	if (Tcl_GetMpzFromObj(interp, objv[1], &b) != TCL_OK ||
	    Tcl_GetMpzFromObj(interp, objv[2], &e) != TCL_OK ||
	    (objc == 4 && Tcl_GetMpzFromObj(interp, objv[3], &m) != TCL_OK))
		return TCL_ERROR;
	result = Tcl_GetObjResult(interp);
	mpz_init(r);
	if (objc == 4)
		mpzerr = mpz_powm(r, b, e, m);
	else
		mpzerr = mpz_pow(r, b, e);
	if (mpzerr != SBN_OK) {
		mpz_clear(r);
		if (mpzerr == SBN_INVAL)
			Tcl_SetStringObj(result, "Negative exponent", -1);
		else
			Tcl_SetStringObj(result, "Out of memory", -1);
		return TCL_ERROR;
	}
	Tcl_SetMpzObj(result, r);
	mpz_clear(r);
	return TCL_OK;
}

static int BigSqrtObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	Tcl_Obj *result;
	int mpzerr;
	mpz_t r; /* result */
	mpz_ptr z; /* input number for the square root */

	if (objc != 2) {
		Tcl_WrongNumArgs(interp, 1, objv, "number");
		return TCL_ERROR;
	}
	if (Tcl_GetMpzFromObj(interp, objv[1], &z) != TCL_OK)
		return TCL_ERROR;
	result = Tcl_GetObjResult(interp);
	mpz_init(r);
	mpzerr = mpz_sqrt(r, z);
	if (mpzerr != SBN_OK) {
		mpz_clear(r);
		Tcl_SetStringObj(result, "Out of memory", -1);
		return TCL_ERROR;
	}
	Tcl_SetMpzObj(result, r);
	mpz_clear(r);
	return TCL_OK;
}
#endif

static int AnnCreateObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	Tcl_Obj *result;
	int *units = alloca(sizeof(int)*objc-1), i;
	struct Ann *net;

	if (objc < 3) {
		Tcl_WrongNumArgs(interp, 1, objv, "OutputUnits ?HiddenUnits1 HiddenUnits2 ...? InputUnits");
		return TCL_ERROR;
	}
	/* Initialize the units vector used to create the net */
	for (i = 1; i < objc; i++) {
		if (Tcl_GetIntFromObj(interp, objv[i], &units[i-1]) != TCL_OK) {
			return TCL_ERROR;
		}
	}
	/* Create the neural net */
	result = Tcl_GetObjResult(interp);
	if ((net = AnnCreateNet(objc-1, units)) == NULL) {
		Tcl_SetStringObj(result, "Out of memory", -1);
		return TCL_ERROR;
	}
	Tcl_SetAnnObj(result, net);
	AnnFree(net);
	return TCL_OK;
}

static int AnnSimulateObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	struct Ann *net;
	Tcl_Obj *varObj, *result;
	int len, j;

	if (objc != 3) {
		Tcl_WrongNumArgs(interp, 1, objv, "AnnVar InputList");
		return TCL_ERROR;
	}
	varObj = Tcl_ObjGetVar2(interp, objv[1], NULL, TCL_LEAVE_ERR_MSG);
	if (!varObj)
		return TCL_ERROR;
	if (Tcl_ListObjLength(interp, objv[2], &len) != TCL_OK)
		return TCL_ERROR;
	/* Get the neural network object */
	if (Tcl_GetAnnFromObj(interp, varObj, &net) != TCL_OK)
		return TCL_ERROR;
	/* Check if the len matches */
	if (len != INPUT_UNITS(net)) {
		Tcl_SetStringObj(Tcl_GetObjResult(interp), "The input list length doesn't match the number of inputs in the neural network", -1);
		return TCL_ERROR;
	}
	/* Set the list elements as the neural net inputs */
	for (j = 0; j < INPUT_UNITS(net); j++) {
		Tcl_Obj *element;
		double d;

		if (Tcl_ListObjIndex(interp, objv[2], j, &element) != TCL_OK)
			return TCL_ERROR;
		if (Tcl_GetDoubleFromObj(interp, element, &d) != TCL_OK)
			return TCL_ERROR;
		INPUT_NODE(net, j) = d;
	}
	/* Simulate! */
	AnnSimulate(net);
	Tcl_InvalidateStringRep(varObj);
	/* Return a list with the output units values */
	result = Tcl_GetObjResult(interp);
	Tcl_SetListObj(result, 0, NULL);
	for (j = 0; j < OUTPUT_UNITS(net); j++) {
		Tcl_Obj *doubleObj;
		doubleObj = Tcl_NewDoubleObj(OUTPUT_NODE(net,j));
		Tcl_ListObjAppendElement(interp, result, doubleObj);
	}
	return TCL_OK;
}

static int AnnConfigureObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	struct Ann *net;
	Tcl_Obj *varObj;
	int j;

	if (objc < 4 || objc % 2) {
		Tcl_WrongNumArgs(interp, 1, objv, "AnnVar Option Value ?Option Value? ...");
		return TCL_ERROR;
	}
	varObj = Tcl_ObjGetVar2(interp, objv[1], NULL, TCL_LEAVE_ERR_MSG);
	if (!varObj)
		return TCL_ERROR;
	/* Get the neural network object */
	if (Tcl_GetAnnFromObj(interp, varObj, &net) != TCL_OK)
		return TCL_ERROR;
	Tcl_InvalidateStringRep(varObj);
	/* process all the option/value pairs */
	for (j = 2; j < objc; j += 2) {
		char *opt = Tcl_GetStringFromObj(objv[j], NULL);
		double dval;

		if (!strcmp(opt, "-learnrate")) {
			if (Tcl_GetDoubleFromObj(interp, objv[j+1], &dval)
			    != TCL_OK)
				return TCL_ERROR;
			LEARN_RATE(net) = dval;
		} else if (!strcmp(opt, "-momentum")) {
			if (Tcl_GetDoubleFromObj(interp, objv[j+1], &dval)
			    != TCL_OK)
				return TCL_ERROR;
			MOMENTUM(net) = dval;
		} else if (!strcmp(opt, "-rpropnminus")) {
			if (Tcl_GetDoubleFromObj(interp, objv[j+1], &dval)
			    != TCL_OK)
				return TCL_ERROR;
			RPROP_NMINUS(net) = dval;
		} else if (!strcmp(opt, "-rpropnplus")) {
			if (Tcl_GetDoubleFromObj(interp, objv[j+1], &dval)
			    != TCL_OK)
				return TCL_ERROR;
			RPROP_NPLUS(net) = dval;
		} else if (!strcmp(opt, "-rpropmaxupdate")) {
			if (Tcl_GetDoubleFromObj(interp, objv[j+1], &dval)
			    != TCL_OK)
				return TCL_ERROR;
			RPROP_MAXUPDATE(net) = dval;
		} else if (!strcmp(opt, "-rpropminupdate")) {
			if (Tcl_GetDoubleFromObj(interp, objv[j+1], &dval)
			    != TCL_OK)
				return TCL_ERROR;
			RPROP_MINUPDATE(net) = dval;
		} else if (!strcmp(opt, "-algo")) {
			char *algo = Tcl_GetStringFromObj(objv[j+1], NULL);
			int algoid = 0;
			if (!strcmp(algo, "rprop")) {
				algoid = ANN_RPROP;
			} else if (!strcmp(algo, "bbprop")) {
				algoid = ANN_BBPROP;
			} else if (!strcmp(algo, "obprop")) {
				algoid = ANN_OBPROP;
			} else if (!strcmp(algo, "bbpropm")) {
				algoid = ANN_BBPROPM;
			} else if (!strcmp(algo, "obpropm")) {
				algoid = ANN_OBPROPM;
			} else {
				Tcl_AppendStringsToObj(Tcl_GetObjResult(interp),
					"unknown algorithm '", algo, "'", NULL);
				return TCL_ERROR;
			}
			AnnSetLearningAlgo(net, algoid);
		} else if (!strcmp(opt, "-scale")) {
			if (Tcl_GetDoubleFromObj(interp, objv[j+1], &dval)
			    != TCL_OK)
				return TCL_ERROR;
			AnnScaleWeights(net, dval);
		} else {
			Tcl_AppendStringsToObj(Tcl_GetObjResult(interp),
				"unknown configuration option '", opt,"'",NULL);
			return TCL_ERROR;
		}
	}
	return TCL_OK;
}

/* ann::train annVar datasetListValue maxEpochs ?maxError? */
static int AnnTrainObjCmd(ClientData clientData, Tcl_Interp *interp,
		int objc, Tcl_Obj *CONST objv[])
{
	struct Ann *net;
	Tcl_Obj *varObj;
	int j, maxepochs, setlen;
	double maxerr = 0, *input = NULL, *target = NULL, *ip, *tp;

	if (objc != 4 && objc != 5) {
		Tcl_WrongNumArgs(interp, 1, objv, "AnnVar DataSetListValue MaxEpochs ?MaxError?");
		return TCL_ERROR;
	}
	varObj = Tcl_ObjGetVar2(interp, objv[1], NULL, TCL_LEAVE_ERR_MSG);
	if (!varObj)
		return TCL_ERROR;
	/* Get the neural network object */
	if (Tcl_GetAnnFromObj(interp, varObj, &net) != TCL_OK)
		return TCL_ERROR;
	Tcl_InvalidateStringRep(varObj);
	/* Extract parameters from Tcl objects */
	if (Tcl_GetIntFromObj(interp, objv[3], &maxepochs) != TCL_OK)
		return TCL_ERROR;
	if (objc == 5 && Tcl_GetDoubleFromObj(interp, objv[4], &maxerr) != TCL_OK)
		return TCL_ERROR;
	if (Tcl_ListObjLength(interp, objv[2], &setlen) != TCL_OK)
		return TCL_ERROR;
	if (setlen % 2) {
		Tcl_SetStringObj(Tcl_GetObjResult(interp), "The dataset list requires an even number of elements", -1);
		return TCL_ERROR;
	}
	/* Convert the dataset from a Tcl list to two C arrays of doubles. */
	ip = input = malloc(INPUT_UNITS(net)*sizeof(double)*(setlen/2));
	tp = target = malloc(OUTPUT_UNITS(net)*sizeof(double)*(setlen/2));
	if (!input || !target) {
		free(input);
		free(target);
		Tcl_SetStringObj(Tcl_GetObjResult(interp),
				"Out of memory in AnnLearnObjCmd()", -1);
		return TCL_ERROR;
	}
	for (j = 0; j < setlen; j++) {
		int l, explen, i;
		Tcl_Obj *sublist;

		if (Tcl_ListObjIndex(interp, objv[2], j, &sublist) != TCL_OK)
			return TCL_ERROR;
		if (Tcl_ListObjLength(interp, sublist, &l) != TCL_OK)
			return TCL_ERROR;
		explen = (j&1) ? OUTPUT_UNITS(net) : INPUT_UNITS(net);
		if (l != explen) {
			free(input);
			free(target);
			Tcl_SetStringObj(Tcl_GetObjResult(interp),
				"Dataset doesn't match input/output units", -1);
			return TCL_ERROR;
		}
		/* Append the data to one of the arrays */
		for (i = 0; i < l; i++) {
			Tcl_Obj *element;
			double t;

			if (Tcl_ListObjIndex(interp, sublist, i, &element)
			    	!= TCL_OK ||
			    Tcl_GetDoubleFromObj(interp, element, &t)
			    	!= TCL_OK)
			{
				free(input);
				free(target);
				return TCL_ERROR;
			}
			if (j&1)
				*tp++ = t;
			else
				*ip++ = t;
		}
	}
	/* Training */
	j = AnnTrain(net, input, target, maxerr, maxepochs, setlen/2);
	free(input);
	free(target);
	Tcl_SetIntObj(Tcl_GetObjResult(interp), j);
	return TCL_OK;
}

/* -------------------------------  Initialization -------------------------- */
int Tclgnegnu_Init(Tcl_Interp *interp)
{
	if (Tcl_InitStubs(interp, "8.0", 0) == NULL)
		return TCL_ERROR;
	if (Tcl_PkgRequire(interp, "Tcl", TCL_VERSION, 0) == NULL)
		return TCL_ERROR;
	if (Tcl_PkgProvide(interp, "tclgnegnu", VERSION) != TCL_OK)
		return TCL_ERROR;
	Tcl_Eval(interp, "namespace eval ann {}");
	Tcl_CreateObjCommand(interp, "ann::create", AnnCreateObjCmd,
			(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
	Tcl_CreateObjCommand(interp, "ann::simulate", AnnSimulateObjCmd,
			(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
	Tcl_CreateObjCommand(interp, "ann::configure", AnnConfigureObjCmd,
			(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
	Tcl_CreateObjCommand(interp, "ann::train", AnnTrainObjCmd,
			(ClientData)NULL, (Tcl_CmdDeleteProc*)NULL);
	/* Private data initialization here */
	return TCL_OK;
}
