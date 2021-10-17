#Games RPS
import sys
sys.path.insert(1, 'D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS')

import pygame
from nnma import *
import cv2
import os
import pickle
import numpy as np
from pygame import Vector2
from tensorflow import keras

os.environ['SDL_Video_CENTERED'] = '1' # Center the window
pygame.init()
pygame.font.init()

keras_model = keras.models.load_model('D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS/Optimizing_RPS/testKeras')
model = Model.load('D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS/Optimizing_RPS/test/Network.model')
RPS_labels = [
	'Rock',
	'Paper',
	'Scissors',
	'Rock'
]

class Game:

	def __init__(self):

		pygame.display.set_caption("My Game")
		self.iconImage = pygame.image.load("icon.png")
		pygame.display.set_icon(self.iconImage)
		self.clock = pygame.time.Clock()
		self.running = True
		self.size = Vector2(800,600)
		self.window = pygame.display.set_mode(Vector2(800,600))
		self.titleFont = pygame.font.SysFont(pygame.font.get_default_font(), 72)
		self.normalFont = pygame.font.SysFont(pygame.font.get_default_font(), 36)
		self.scene = 1
		self.mbut = (False, False, False)
		self.object = 4
		self.model_use = "M"
		
		self.buttons = [Button(self, (self.width/2,250), text="Play", mode="CENTER", font=self.normalFont)

		]

	def processInput(self):

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
				break
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					self.running = False
					break

		self.last_mbut = self.mbut
		self.mpos = pygame.mouse.get_pos() # For button (if on it => on it = on)
		self.mbut = pygame.mouse.get_pressed()

		for button in self.buttons:
			if button.on_it():
				button.on = True
			else:
				button.on = False

		if self.scene == 3:
			#cv2.namedWindow("Preview")
			vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

			if vc.isOpened(): # Try to get the first frame
				rval, self.frame = vc.read()
			else:
				rval = False

			rval, self.frame = vc.read()
			cv2.imshow("Preview", self.frame)
			self.frame = cv2.resize(self.frame, (28,28))
			cv2.imwrite('cam/im.png', self.frame)
			
		
	def update(self):

		if self.scene == 3:
			image_datas = []
			data = cv2.imread('cam/im.png', cv2.IMREAD_GRAYSCALE)

			if self.model_use == "My":
				image_data = (data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
				self.confidences = model.predict(image_data)
				predictions = model.output_layer_activation.predictions(self.confidences)
				self.object_class = predictions[0]
				self.object = RPS_labels[predictions[0]]
			else:
				image_data = np.expand_dims(data, -1)
				image_data = np.expand_dims(image_data,0)
				image_data = image_data.astype('float16')
				image_datas.append(image_data)
				image_datas.append(image_data)
				image_datas = np.array(image_datas)
				self.confidences = keras_model.predict(image_data)
				self.object_class = np.argmax(self.confidences)
				self.object = RPS_labels[self.object_class]

			print(self.confidences)
			print(self.object_class, self.object)

		
		if self.buttons[0].on and self.mbut[0] and (not self.last_mbut[0]):
			self.scene = 2


	def text(self, surface, text, pos, font, col, mode="LEFT", antialias=True, background=None):
		if mode=="CENTER":
			textSurface = font.render(str(text), antialias, col, background)
			size = font.size(text)
			surface.blit(textSurface, (pos[0] - size[0]/2, pos[1] - size[1]/2))
			return size
		elif mode=="LEFT":
			text = font.render(str(text), antialias, col, background)
			surface.blit(text, pos)
			return size
		elif mode=="RIGHT":
			textSurface = font.render(str(text), antialias, col, background)
			size = font.size(text)
			surface.blit(textSurface, (pos[0] - size[0], pos[1] - size[1]))
			return size
		else:
			print("Not a valid Mode")


	def render_menu(self):
		sizePlay = self.normalFont.size("Play")
		if self.buttons[0].on:
			back = (200,200,200)
		else: 
			back = None

		self.buttons[0].render(back=back)
		self.text(self.window, "Rock, Paper, Scissors", (self.width/2,80), self.titleFont, (0,0,0), mode="CENTER" )
		 
	def render_animationStart(self):
		self.text(self.window, "Ready?", (self.width/2,self.height/2), self.titleFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(2000)
		self.window.fill((255,255,255))
		self.text(self.window, "Rock", (self.width/2,self.height/2), self.titleFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(1000)
		self.window.fill((255,255,255))
		self.text(self.window, "Paper", (self.width/2,self.height/2), self.titleFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(1000)
		self.window.fill((255,255,255))
		self.text(self.window, "Scissors", (self.width/2,self.height/2), self.titleFont, (0,0,0), mode="CENTER" )
		pygame.display.update()
		pygame.time.delay(500)
		self.scene = 3

	def render_play(self):
		if self.object_class != 4:
			answer = RPS_labels[self.object_class + 1]
			self.text(self.window, str(answer), (self.width/2,self.height/2), self.titleFont, (0,0,0), mode="CENTER" )
			pygame.display.update()
			pygame.time.delay(8000)
			self.scene = 1

	def render(self):

		self.window.fill((255,255,255))
		if self.scene == 1:
			self.render_menu()
		elif self.scene == 2:
			self.render_animationStart()
		elif self.scene == 3:
			self.render_play()
		pygame.display.update()

	def run(self):

		while self.running:
			self.processInput()
			self.update()
			self.render()
			self.clock.tick(60)

	@property
	def width(self):
		return int(self.size.x)

	@property
	def height(self):
		return int(self.size.y)

class Button():

	def __init__(self, interface, pos, width=None, height=None, text=None, mode="LEFT", font=pygame.font.SysFont(pygame.font.get_default_font(), 36)):
		self.interface = interface
		self.mode = mode
		self.pos = pos
		self.x = pos[0]
		self.y = pos[1]
		self.width = width
		self.height = height
		self.text = text
		self.font = font
		self.on = False

	def on_it(self):
		mpos = pygame.mouse.get_pos()
		sizeText = self.font.size(self.text)
		if self.mode=="CENTER":
			if self.x - sizeText[0]/2 < mpos[0] and \
			   self.x + sizeText[0]/2 > mpos[0] and \
			   self.y - sizeText[1]/2 < mpos[1] and \
			   self.y + sizeText[1]/2 > mpos[1]:
				return True
			else:
				return False
		elif mode=="LEFT":
			if self.x < self.mpos[0] and \
			   self.x + sizeText[1] > mpos[0] and \
			   self.y < self.mpos[1] and \
			   self.y + sizeText[1] > mpos[1]:
				return True
			else:
				return False
		elif mode=="RIGHT":
			if self.x - sizeText[0] < mpos[0] and \
			   self.x > self.mpos[0] and \
			   self.y - sizeText[1] < mpos[1] and \
			   self.y > self.mpos[1]:
				return True
			else:
				return False

	def render(self, col=(0,0,0), back=None):
		self.interface.text(self.interface.window, self.text, self.pos, self.font, col, mode=self.mode, background=back)



game = Game()
game.run()

