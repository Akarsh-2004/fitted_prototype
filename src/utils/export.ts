import html2canvas from 'html2canvas'

export const exportBoardPng = async (element: HTMLElement, name: string) => {
  const canvas = await html2canvas(element, { useCORS: true, scale: 2, backgroundColor: null, logging: false })
  const a = document.createElement('a')
  a.download = `${name || 'vestir-board'}.png`
  a.href = canvas.toDataURL('image/png')
  a.click()
}
