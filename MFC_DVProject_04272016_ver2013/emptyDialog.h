#pragma once


// emptyDialog dialog

class emptyDialog : public CDialogEx
{
	DECLARE_DYNAMIC(emptyDialog)

public:
	emptyDialog(CWnd* pParent = NULL);   // standard constructor
	virtual ~emptyDialog();

// Dialog Data
	enum { IDD = IDD_DIALOG1 };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
};
